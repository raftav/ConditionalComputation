from __future__ import division
import tensorflow as tf
import skip_rnn_cells

from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import init_ops

from tensorflow.python.util import nest

def variable_summaries(var,var_name):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope(var_name+'_summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


class Model(object):

	# layer normalization
	def _norm(self,inp, scope , norm_gain=1.0, norm_shift=0.0):
		shape = inp.get_shape()[-1:]
		gamma_init = init_ops.constant_initializer(norm_gain)
		beta_init = init_ops.constant_initializer(norm_shift)

		with vs.variable_scope(scope):
			# Initialize beta and gamma for use by layer_norm.
			vs.get_variable("gamma", shape=shape, initializer=gamma_init)
			vs.get_variable("beta", shape=shape, initializer=beta_init)

		normalized = layers.layer_norm(inp, reuse=True, scope=scope)
		return normalized

	def __init__(self,features,labels,seq_length,config,is_training):

		# batch size cannot be inferred from features shape because
		# it must be defined statically
		if is_training:
			batch_size=config.batch_size
		else:
			batch_size=config.num_examples_val

		# global step for learning rate decay
		global_step = tf.Variable(0,name='global_step', trainable=False)
		self._global_step=global_step

		# slope of the sigmoid for slope annealing trick
		slope = tf.to_float(global_step / config.updating_step) * tf.constant(config.slope_annealing_rate) + tf.constant(1.0)
		self._slope = slope

		# stack of custom rnn cells
		num_units = [config.n_hidden for _ in range(config.num_layers)]

		with tf.variable_scope('forward_cells'):
			multi_cell_fw = skip_rnn_cells.MultiSkipGRUCell(num_units,layer_norm=True)
			initial_state_fw = multi_cell_fw.trainable_initial_state(batch_size)

		with tf.variable_scope('backward_cells'):
			multi_cell_bw = skip_rnn_cells.MultiSkipGRUCell(num_units,layer_norm=True)
			initial_state_bw = multi_cell_fw.trainable_initial_state(batch_size)

		# linear mapping of features dimension to dimension of
		# first hidden layer
		with tf.variable_scope('embedding'):
			embedding_weights = tf.get_variable('embedding_weights',
								[config.audio_feat_dimension,config.n_hidden],
								initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))
			
			features = tf.reshape(features,[-1,config.audio_feat_dimension])

			embedded_input = tf.matmul(features,embedding_weights)

			embedded_input = self._norm(embedded_input,"input")
			embedded_input = tf.identity(embedded_input,name='embedding_rescaled')

			embedded_input = tf.reshape(embedded_input,[batch_size,-1,config.n_hidden])


		with tf.variable_scope('forward_rnn'):
			rnn_outputs , last_state_fw  = tf.nn.dynamic_rnn(multi_cell_fw,
															embedded_input,
															initial_state=initial_state_fw,
															sequence_length=seq_length)
			
			rnn_outputs_fw, updated_states_fw = rnn_outputs.h, rnn_outputs.state_gate



			with tf.variable_scope('backward_rnn'):
				input_reverse = tf.reverse_sequence(input=embedded_input,
												seq_lengths=seq_length,
												seq_axis=1, batch_axis=0)

				rnn_outputs , last_state_fw  = tf.nn.dynamic_rnn(multi_cell_bw,
																input_reverse,
																initial_state=initial_state_bw,
																sequence_length=seq_length)
				
				rnn_outputs_bw, updated_states_bw = rnn_outputs.h, rnn_outputs.state_gate

				rnn_outputs_bw = tf.reverse_sequence(input=rnn_outputs_bw,
												seq_lengths=seq_length,
												seq_axis=1, batch_axis=0)
				updated_states_bw = tf.reverse_sequence(input=updated_states_bw,
												seq_lengths=seq_length,
												seq_axis=1, batch_axis=0)


		with tf.variable_scope("Output"):
			
			output_fw_weights = tf.get_variable('forward_weights',[config.n_hidden,config.num_classes],dtype=tf.float32,
			                        initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))
			output_bw_weights = tf.get_variable('backward_weights',[config.n_hidden,config.num_classes],dtype=tf.float32,
			                        initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))
			output_biases = tf.get_variable('biases',shape=[config.num_classes],dtype=tf.float32,
											initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))

			rnn_outputs_fw = tf.reshape(rnn_outputs_fw,[-1,config.n_hidden])
			rnn_outputs_bw = tf.reshape(rnn_outputs_bw,[-1,config.n_hidden])		
			
			output = tf.matmul(rnn_outputs_fw,output_fw_weights) + tf.matmul(rnn_outputs_bw,output_bw_weights) + output_biases
		
			logits = tf.reshape(output,[batch_size,-1,config.num_classes])


		if is_training:

			# evaluate cost and optimize
			with tf.name_scope('cost'):

				all_states_fw=tf.reduce_sum(updated_states_fw)
				all_states_bw=tf.reduce_sum(updated_states_bw)
				all_states=all_states_bw+all_states_fw

				cross_entropy_loss = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels))
				l2_loss = config.lambda_l2 * all_states
				loss = cross_entropy_loss + l2_loss

				#self._cross_entropy_loss = cross_entropy_loss
				#self._l2_loss = l2_loss
				self._cost = loss

				tf.summary.scalar('cost',self._cost)

			with tf.name_scope('optimizer'):
				learning_rate = tf.train.exponential_decay(config.learning_rate, global_step,config.updating_step, config.learning_decay, staircase=True)
				self._learning_rate= learning_rate

				if 'momentum' in config.optimizer_choice:
					self._optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)

				elif 'adam' in config.optimizer_choice:
					self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

				# gradient clipping
				gradients , variables = zip(*self._optimizer.compute_gradients(self._cost))
				clip_grad  = [tf.clip_by_norm(gradient, 1.0) for gradient in gradients] 
				self._optimize = self._optimizer.apply_gradients(zip(clip_grad,variables),global_step=self._global_step)

		else:

			binary_mask = tf.sequence_mask(seq_length)
			prediction=tf.cast(tf.argmax(logits, axis=2),tf.int32)
			masked_prediction = tf.boolean_mask(prediction,binary_mask)
			masked_labels = tf.boolean_mask(labels,binary_mask)

			correct = tf.equal(masked_prediction,masked_labels)
			self._accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))

			self._prediction = prediction
			self._labels = labels

			states_fw={}
			states_bw={}

			i=config.num_layers

			updated_states_fw = tf.boolean_mask(updated_states_fw,binary_mask)
			updated_states_bw = tf.boolean_mask(updated_states_bw,binary_mask)

			states_fw['z_{:d}'.format(i)] = updated_states_fw
			states_bw['z_{:d}'.format(i)] = updated_states_bw

			self._binary_states_fw = states_fw

			self._binary_states_bw = states_bw

	@property
	def cost(self):
		return self._cost

	@property
	def optimize(self):
		return self._optimize

	@property
	def cell_slope(self):
		return self._cell_slope

	@property
	def prediction(self):
		return self._prediction

	@property
	def accuracy(self):
		return self._accuracy

		
	@property
	def binary_states_fw(self):
		return self._binary_states_fw

		
	@property
	def binary_states_bw(self):
		return self._binary_states_bw


	@property
	def labels(self):
		return self._labels

	@property
	def learning_rate(self):
		return self._learning_rate

	@property
	def global_step(self):
		return self._global_step

	@property
	def slope(self):
		return self._slope

	@property
	def decoded(self):
		return self._decoded
from __future__ import division
import tensorflow as tf
import sys

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

	def __init__(self,features,labels,seq_length,lab_length,config,is_training):

		batch_size = tf.shape(features)[0]
		
		global_step = tf.Variable(0, trainable=False)
		self._global_step=global_step

		with tf.variable_scope('forward'):

			forward_cells = []
			for i in range(config.num_layers):
				with tf.variable_scope('layer_{:d}'.format(i)):
					lstm_cell_forward = tf.contrib.rnn.GRUCell(config.n_hidden,
											activation=tf.tanh,
                                            kernel_initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1),
                                            bias_initializer=tf.ones_initializer())

					forward_cells.append(lstm_cell_forward)

			#initial_states_fw=forward_cells.zero_state(batch_size,tf.float32)

		with tf.variable_scope('backward'):

			backward_cells = []
			for i in range(config.num_layers):
				with tf.variable_scope('layer_{:d}'.format(i)):
					lstm_cell_backward = tf.contrib.rnn.GRUCell(config.n_hidden,
											activation=tf.tanh,
                                            kernel_initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1),
                                            bias_initializer=tf.ones_initializer())
					
					backward_cells.append(lstm_cell_backward)

			#initial_states_bw=forward_cells.zero_state(batch_size,tf.float32)

		with tf.variable_scope('RNN'):
			rnn_outputs, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
														cells_fw=forward_cells,
														cells_bw=backward_cells,
														inputs=features,
														initial_states_fw=None,
														initial_states_bw=None,
														dtype=tf.float32,
														sequence_length=seq_length,
														parallel_iterations=None,
														scope=None)

			rnn_outputs_fw , rnn_outputs_bw = tf.split(rnn_outputs,num_or_size_splits=2, axis=2)


		output_weights_fw = tf.get_variable('outputs_weights_fw',dtype=tf.float32,
							shape=[config.n_hidden,config.num_classes],
							initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))
		
		output_weights_bw = tf.get_variable('outputs_weights_bw',dtype=tf.float32,
							shape=[config.n_hidden,config.num_classes],
							initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))

		output_biases = tf.get_variable('biases',dtype=tf.float32,
										shape=[config.num_classes],
										initializer=tf.random_uniform_initializer(minval=-1.0,maxval=1.0))
										#initializer=tf.constant(5.0,shape=[config.num_classes]))
										
		rnn_outputs_fw = tf.reshape(rnn_outputs_fw,[-1,config.n_hidden])
		rnn_outputs_bw = tf.reshape(rnn_outputs_bw,[-1,config.n_hidden])


		logits = tf.matmul(rnn_outputs_fw,output_weights_fw) + tf.matmul(rnn_outputs_bw,output_weights_bw) + output_biases
		logits = tf.reshape(logits,[batch_size,-1,config.num_classes])

		# ctc operations are defined as time_major
		logits = tf.transpose(logits, (1, 0, 2))
		#logits =tf.add(logits,tf.constant(1.0e-8,dtype=tf.float32))

		# trasform target labels into sparse tensor for the ctc loss
		# indices : little hack to get the indices in the proper format.
		# there is no place where a label is equal to num_classes + 1

		# the following code gives problem when the batch size is bigger than one.
		# In this case I cannot distinguish between true "0" labels and 0 labels
		# coming from the padding.
		# Something strange then happens to the loss calculation, which is difficult to underdstand.
		# Anyway, it results in infinite loss.
		# 
		sparse_labels = tf.contrib.keras.backend.ctc_label_dense_to_sparse(labels,lab_length)
		dense_labels = tf.sparse_tensor_to_dense(sparse_labels)


		#tf.assert_equal(labels,dense_labels)

		if is_training:
			# evaluate cost and optimize
			with tf.name_scope('cost'):
				loss = tf.nn.ctc_loss(tf.to_int32(sparse_labels),logits,sequence_length=seq_length,
									preprocess_collapse_repeated=False,ctc_merge_repeated=True,time_major=True)

				self._cost = tf.reduce_mean( loss )
				tf.summary.scalar('cost',self._cost)

			with tf.name_scope('optimizer'):
				learning_rate = tf.train.exponential_decay(config.learning_rate, global_step,
			                        config.updating_step, config.learning_decay, staircase=True)

				self._learning_rate= learning_rate

				if "momentum" in config.optimizer_choice:
					self._optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)
				elif "adam" in config.optimizer_choice:
					self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
				else:
					print("Optimizer must be either momentum or adam. Closing.")
					sys.exit()
					
				gradients , variables = zip(*self._optimizer.compute_gradients(self._cost))
				clip_grad  = [tf.clip_by_norm(gradient, 1.0) for gradient in gradients] 
				self._optimize = self._optimizer.apply_gradients(zip(clip_grad,variables),global_step=self._global_step)
		
		else:
			# Option 2: tf.nn.ctc_beam_search_decoder
			# (it's slower but you'll get better results)
			#self._decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_length)
			self._decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_length)

			self._sparse_labels = sparse_labels
			

	@property
	def cost(self):
		return self._cost

	@property
	def optimize(self):
		return self._optimize

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
	def ler(self):
		return self._ler

	@property
	def decoded(self):
		return self._decoded

	@property
	def dense_decoded(self):
		return self._dense_decoded

	@property
	def sparse_labels(self):
		return self._sparse_labels
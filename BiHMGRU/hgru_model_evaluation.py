from __future__ import division
import tensorflow as tf
import hgru

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

		batch_size=1

		# global step for learning rate decay
		global_step = tf.Variable(0,name='global_step', trainable=False)
		self._global_step=global_step

		# slope of the sigmoid for slope annealing trick
		slope = tf.to_float(global_step / config.updating_step) * tf.constant(config.slope_annealing_rate) + tf.constant(1.0)
		self._slope = slope

		# stack of custom rnn cells
		with tf.variable_scope('forward_cells'):
			cell_list_fw=[]
			zero_states_fw = []

			for i in range(config.num_layers):
				with tf.variable_scope('layer_{:d}'.format(i)):
					rnn_cell= hgru.HGCell(config.n_hidden)
					rnn_cell.slope=slope

					h_init=tf.get_variable('h_init_state',[1,rnn_cell.state_size.h],
											initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1),
											dtype=tf.float32)
					h_init=tf.tile(h_init,[batch_size,1])

					z_init=tf.ones([1,rnn_cell.state_size.z])
					z_init=tf.tile(z_init,[batch_size,1])


				zero_states_fw.append(hgru.HardGatedStateTuple(h_init,z_init))

				cell_list_fw.append(rnn_cell)

			self._cell_slope = cell_list_fw[0].slope

			multi_cell_fw = hgru.MultiHGRNNCell(cell_list_fw)

			initial_state_fw=tuple(zero_states_fw)
	
		with tf.variable_scope('backward_cells'):
			cell_list_bw=[]
			zero_states_bw = []

			for i in range(config.num_layers):
				with tf.variable_scope('layer_{:d}'.format(i)):
					rnn_cell= hgru.HGCell(config.n_hidden)
					rnn_cell.slope=slope

					h_init=tf.get_variable('h_init_state',[1,rnn_cell.state_size.h],
											initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1),
											dtype=tf.float32)
					h_init=tf.tile(h_init,[batch_size,1])

					z_init=tf.ones([1,rnn_cell.state_size.z])
					z_init=tf.tile(z_init,[batch_size,1])


				zero_states_bw.append(hgru.HardGatedStateTuple(h_init,z_init))

				cell_list_bw.append(rnn_cell)


			multi_cell_bw = hgru.MultiHGRNNCell(cell_list_bw)

			initial_state_bw=tuple(zero_states_bw)


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


		# FORWARD RNN
		# use dynamic_rnn for training
		with tf.variable_scope('forward_rnn'):
			rnn_outputs_fw, last_state_fw = tf.nn.dynamic_rnn(multi_cell_fw,embedded_input,
														sequence_length=seq_length,
														initial_state=initial_state_fw)

		rnn_hidden_layers_fw = []
		rnn_binary_states_fw = []
		rnn_binary_logits_fw = []
		for i in range(config.num_layers):
			rnn_hidden_layers_fw.append(rnn_outputs_fw[i].h)
			rnn_binary_states_fw.append(rnn_outputs_fw[i].z)
			rnn_binary_logits_fw.append(rnn_outputs_fw[i].z_tilda)

		rnn_outputs_fw = rnn_hidden_layers_fw
				

		with tf.variable_scope("Output_FW"):
			
			# output is another neural network with input from all hidden layers
			with tf.device('/gpu:0'):
				# concatenate rnn_ouputs along last dimension.
				# will get to [batch_size,max_time,num_layers*num_hidden_units]
				concat_rnn_output_fw = tf.concat(rnn_outputs_fw,axis=2)

				# flatten batch and time dimension of concatenate output
				concat_rnn_output_fw = tf.reshape(concat_rnn_output_fw,shape=[-1,concat_rnn_output_fw.get_shape().as_list()[2]])

				output_logits_fw = []

				for i in range(config.num_layers):

					#flatten batch and time dimensions of the layer i outputs
					rnn_outputs_layer_fw = tf.reshape(rnn_outputs_fw[i],shape=[-1,rnn_outputs_fw[i].get_shape().as_list()[2]])

					# W_l^e
					output_embedding_matrix_fw = tf.get_variable("W{}".format(i),[ rnn_outputs_layer_fw.get_shape().as_list()[1] , config.num_classes ],
												dtype=tf.float32,
												initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))

					
					# output from a single layer
					output_logit_fw = tf.matmul(rnn_outputs_layer_fw, output_embedding_matrix_fw)
					

					output_logits_fw.append(output_logit_fw)


		# BACKWARD RNN
		input_reverse = tf.reverse_sequence(input=embedded_input,
												seq_lengths=seq_length,
												seq_axis=1, batch_axis=0)

		with tf.variable_scope('backward_rnn'):
			rnn_outputs_bw, last_state_bw = tf.nn.dynamic_rnn(multi_cell_bw,input_reverse,
															sequence_length=seq_length,
															initial_state=initial_state_bw)


		rnn_hidden_layers_bw = []
		rnn_binary_states_bw = []
		rnn_binary_logits_bw = []

		for i in range(config.num_layers):
			# reverse all outputs before appending
			# rnn_output is a list (len=num_hidden_layer) of named tuple (h,z,z_tilda).

			temp_h = rnn_outputs_bw[i].h
			temp_h = tf.reverse_sequence(input=temp_h,
												seq_lengths=seq_length,
												seq_axis=1, batch_axis=0)
			temp_z = rnn_outputs_bw[i].z
			temp_z = tf.reverse_sequence(input=temp_z,
												seq_lengths=seq_length,
												seq_axis=1, batch_axis=0)
			temp_z_tilda=rnn_outputs_bw[i].z_tilda
			temp_z_tilda = tf.reverse_sequence(input=temp_z_tilda,
												seq_lengths=seq_length,
												seq_axis=1, batch_axis=0)


			rnn_hidden_layers_bw.append(temp_h)
			rnn_binary_states_bw.append(temp_z)
			rnn_binary_logits_bw.append(temp_z_tilda)

		rnn_outputs_bw = rnn_hidden_layers_bw


		with tf.variable_scope("Output_BW"):
			
			# output is another neural network with input from all hidden layers
			with tf.device('/gpu:0'):
				# concatenate rnn_ouputs along last dimension.
				# will get to [batch_size,max_time,num_layers*num_hidden_units]
				concat_rnn_output_bw = tf.concat(rnn_outputs_bw,axis=2)

				# flatten batch and time dimension of concatenate output
				concat_rnn_output_bw = tf.reshape(concat_rnn_output_bw,shape=[-1,concat_rnn_output_bw.get_shape().as_list()[2]])


				output_logits_bw = []


				for i in range(config.num_layers):

					#flatten batch and time dimensions of the layer i outputs
					rnn_outputs_layer_bw = tf.reshape(rnn_outputs_bw[i],shape=[-1,rnn_outputs_bw[i].get_shape().as_list()[2]])

					# W_l^e
					output_embedding_matrix_bw = tf.get_variable("W{}".format(i),[ rnn_outputs_layer_bw.get_shape().as_list()[1] , config.num_classes ],
												dtype=tf.float32,
												initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))

					
					# output from a single layer
					output_logit_bw = tf.matmul(rnn_outputs_layer_bw, output_embedding_matrix_bw)
					

					output_logits_bw.append(output_logit_bw)

				# Add independent forward and backward outputs
				output_logits=[]
				for i in range(config.num_layers):
					output_logits.append( output_logits_bw[i] + output_logits_fw[i] )

				# combined output from all layers
				output_logits = self._norm(tf.add_n(output_logits),"output_logits")
				output = tf.nn.relu(output_logits)
				
				# shape back to [batch_size, max_time, num_classes]
				logits = tf.reshape(output,shape=[batch_size,-1,config.num_classes])



		prediction=tf.cast(tf.argmax(logits, axis=2),tf.int32)

		correct = tf.equal(prediction,labels)
		self._accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))

		self._prediction = prediction
		self._labels = labels

		activation_fw={}
		states_fw={}
		states_logit_fw={}

		for i in range(config.num_layers):
			states_fw['z_{:d}'.format(i)] = rnn_binary_states_fw[i]
			states_logit_fw['z_tilda_{:d}'.format(i)] = rnn_binary_logits_fw[i]
			activation_fw['{}'.format(i)] = tf.norm(rnn_outputs_fw[i],ord=1,axis=2)

		activation_bw={}
		states_bw={}
		states_logit_bw={}

		for i in range(config.num_layers):
			states_bw['z_{:d}'.format(i)] = rnn_binary_states_bw[i]
			states_logit_bw['z_tilda_{:d}'.format(i)] = rnn_binary_logits_bw[i]
			activation_bw['{}'.format(i)] = tf.norm(rnn_outputs_bw[i],ord=1,axis=2)


		posteriors=tf.nn.softmax(logits)
		self._posteriors=posteriors

		self._binary_states_fw = states_fw
		self._binary_logits_fw = states_logit_fw
		self._activations_norm_fw = activation_fw

		self._binary_states_bw = states_bw
		self._binary_logits_bw = states_logit_bw
		self._activations_norm_bw = activation_bw


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
	def posteriors(self):
		return self._posteriors

	@property
	def accuracy(self):
		return self._accuracy

		
	@property
	def binary_states_fw(self):
		return self._binary_states_fw

	@property
	def binary_logits_fw(self):
		return self._binary_logits_fw

	@property
	def activations_norm_fw(self):
		return self._activations_norm_fw

		
	@property
	def binary_states_bw(self):
		return self._binary_states_bw

	@property
	def binary_logits_bw(self):
		return self._binary_logits_bw

	@property
	def activations_norm_bw(self):
		return self._activations_norm_bw

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
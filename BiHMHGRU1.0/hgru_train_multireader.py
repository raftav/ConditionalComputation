from __future__ import print_function
from __future__ import division

# Avoid printing tensorflow log messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
import time
import sys
import pickle

import hgru_model

#Set verbosity to only show errors (no Info or Warnings)
tf.logging.set_verbosity(tf.logging.WARN)

#################################
# Useful constant and paths
#################################
ExpNum = int(sys.argv[1])

num_examples=3696

class Configuration(object):
	
	learning_rate=float(sys.argv[2])
	slope_annealing_rate=float(sys.argv[3])
	updating_step=int(sys.argv[4])
	learning_decay=float(sys.argv[5])
	keep_prob=float(sys.argv[6])
	batch_size=int(sys.argv[7])
	optimizer_choice=sys.argv[8]

	lambda_l2 = float(sys.argv[9])

	audio_feat_dimension = 120

	num_classes = 144

	num_examples_val=192

	num_epochs=5000
	
	n_hidden=250
	num_layers=5


checkpoints_dir='checkpoints/exp'+str(ExpNum)+'/'

tensorboard_dir='tensorboard/exp'+str(ExpNum)+'/'

z_states_dir='z_states/exp'+str(ExpNum)+'/'

z_hist_dir='z_hist/exp'+str(ExpNum)+'/'

traininglog_dir='training_logs/'


if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)

if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)

if not os.path.exists(z_states_dir):
    os.makedirs(z_states_dir)

if not os.path.exists(z_hist_dir):
    os.makedirs(z_hist_dir)

if not os.path.exists(traininglog_dir):
	os.makedirs(traininglog_dir)

trainingLogFile=open(traininglog_dir+'TrainingExperiment'+str(ExpNum)+'.txt','w')

testLogFile=open(traininglog_dir+'TestResultsExperiment'+str(ExpNum)+'.txt','w')

###################################
# Auxiliary functions
###################################


#########################################
# Input pipelines
#########################################

# Reads a single serialized SequenceExample
def read_my_file_format(filename_queue,feat_dimension=123):

    reader = tf.TFRecordReader()

    key, serialized_example = reader.read(filename_queue)

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(serialized_example,
                                               context_features={"length": tf.FixedLenFeature([],dtype=tf.int64)},
                                               sequence_features={"audio_feat":tf.FixedLenSequenceFeature([feat_dimension],dtype=tf.float32),
                                                                  "audio_labels":tf.FixedLenSequenceFeature([],dtype=tf.float32)}
                                        )

    return tuple((context_parsed['length'],sequence_parsed['audio_feat'],tf.to_int32(sequence_parsed['audio_labels'])))

# training input pipeline
def input_pipeline(filenames,config):

    filename_queue = tf.train.string_input_producer(filenames, num_epochs=config.num_epochs, shuffle=True)
    
    example_list = [read_my_file_format(filename_queue,feat_dimension=config.audio_feat_dimension) 
    				for _ in range(10)]

    seq_length_batch, audio_features_batch, audio_labels_batch  = tf.train.batch_join(example_list,
                                                    batch_size=config.batch_size,
                                                    capacity=100,
                                                    dynamic_pad=True,
                                                    enqueue_many=False)


    return audio_features_batch, audio_labels_batch, seq_length_batch


def input_pipeline_validation(filenames,config):

    filename_queue = tf.train.string_input_producer(filenames, num_epochs=None, shuffle=False)
    
    example_list = [read_my_file_format(filename_queue,feat_dimension=config.audio_feat_dimension) for _ in range(10)]

    seq_length_batch, audio_features_batch, audio_labels_batch  = tf.train.batch_join(example_list,
                                                    batch_size=config.num_examples_val,
                                                    capacity=100,
                                                    dynamic_pad=True,
                                                    enqueue_many=False)


    return audio_features_batch, audio_labels_batch, seq_length_batch

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

#################################
# Training module
#################################

def train():

	config=Configuration()

	# list of input filenames + check existence
	filename_train=['/home/local/IIT/rtavarone/Data/PreProcessKaldiTimit/TRAIN/sequence_full_{:04d}.tfrecords'.format(i) \
					 for i in range(num_examples)]
	for f in filename_train:
		if not tf.gfile.Exists(f):
			raise ValueError('Failed to find file: ' + f)

	# list of input filenames + check existence
	filename_val=['/home/local/IIT/rtavarone/Data/PreProcessKaldiTimit/CORETEST/sequence_full_{:04d}.tfrecords'.format(i) \
					 for i in range(config.num_examples_val)]
	for f in filename_train:
		if not tf.gfile.Exists(f):
			raise ValueError('Failed to find file: ' + f)

	# training graph		
	with tf.Graph().as_default():

		# extract batch examples
		with tf.device('/cpu:0'):
			with tf.name_scope('train_batch'):
				audio_features, audio_labels, seq_length = input_pipeline(filename_train,config)

			with tf.name_scope('validation_batch'):
				audio_features_val, audio_labels_val, seq_length_val = input_pipeline_validation(filename_val,config)

		# audio features reconstruction
		with tf.device('/cpu:0'):
			with tf.variable_scope('model',reuse=None):
				print('Building training model:')
				start=time.time()
				train_model = hgru_model.Model(audio_features,audio_labels,seq_length,config,is_training=True)
				print('Graph creation time = ',time.time() - start)
				print('done.\n')

			with tf.variable_scope('model',reuse=True):
				print('Building validation model:')
				start=time.time()
				val_model = hgru_model.Model(audio_features_val,audio_labels_val,seq_length_val,config,is_training=None)
				print('Graph creation time = ',time.time() - start)
				print('done.')

			# variables initializer
			print('Initializer creation')
			start=time.time()
			init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
			print('creation time = ',time.time() - start)
			print('Done')

		# save and restore all the variables.
		saver = tf.train.Saver(max_to_keep=10)

		# print trainable variables
		#print('')
		#print('TRAINABLE VARIABLES:')
		
		variables = [f for f in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
		for var in variables:
			#print(var.name)
			name=var.name.replace(':0','')
			variable_summaries(var,name)
		#print('')

		# tensorboard merger
		merged_summary_op = tf.summary.merge_all()

		# start session
		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
											  log_device_placement=False)) as sess:

			# run initializer
			print('Initializing variable')
			start=time.time()
			sess.run(init_op)
			print('initialization time = ',time.time() - start)
			print('Done')

			# start queue coordinator and runners
			print('Starting coordinators')
			start=time.time()
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(sess = sess, coord = coord)
			print('time = ',time.time() - start)
			print('Done')

			# print experiment info to stdout
			print('')
			print('## EXPERIMENT NUMBER ',ExpNum)
			trainingLogFile.write('# EXPERIMENT NUMBER {:d} \n'.format(ExpNum))

			print('## binary units : deterministic')
			trainingLogFile.write('# binary units : deterministic \n')
			
			print('## optimizer : ',config.optimizer_choice)
			trainingLogFile.write('## optimizer : {:s} \n'.format(config.optimizer_choice))
			
			print('## number of hidden layers : ',config.num_layers)
			trainingLogFile.write('## number of hidden layers : {:d} \n'.format(config.num_layers))
			
			print('## number of hidden units : ',config.n_hidden)
			trainingLogFile.write('## number of hidden units : {:d} \n'.format(config.n_hidden))
			
			print('## learning rate : ',config.learning_rate)
			trainingLogFile.write('## learning rate : {:.6f} \n'.format(config.learning_rate))
			
			print('## learning rate update steps: ',config.updating_step)
			trainingLogFile.write('## learning rate update steps: {:d} \n'.format(config.updating_step))
			
			print('## learning rate decay : ',config.learning_decay)
			trainingLogFile.write('## learning rate decay : {:.6f} \n'.format(config.learning_decay))
			
			print('## slope_annealing_rate : ',config.slope_annealing_rate)
			trainingLogFile.write('## slope_annealing_rate : {:.6f} \n'.format(config.slope_annealing_rate))
			
			print('## dropout keep prob (no dropout if 1.0) :',config.keep_prob)
			trainingLogFile.write('## dropout keep prob (no dropout if 1.0) : {:.6f} \n'.format(config.keep_prob))
			
			print('## batch size : ',config.batch_size)
			trainingLogFile.write('## batch size : {:d} \n'.format(config.batch_size))

			print('## lambda l2 : ',config.lambda_l2)
			trainingLogFile.write('## lambda l2 : {:f} \n'.format(config.lambda_l2))			
			
			print('## number of steps: ',num_examples*config.num_epochs/config.batch_size)
			trainingLogFile.write('## approx number of steps: {:d} \n'.format(int(num_examples*config.num_epochs/config.batch_size)))
			
			print('## number of steps per epoch: ',num_examples/config.batch_size)
			trainingLogFile.write('## approx number of steps per epoch: {:d} \n'.format(int(num_examples/config.batch_size)))
			print('')

			trainingLogFile.flush()

			# op to write logs to Tensorboard
			summary_writer = tf.summary.FileWriter(tensorboard_dir, graph=tf.get_default_graph())

			try:
				epoch_counter=1
				epoch_cost=0.0
				EpochStartTime=time.time()
				partial_time=time.time()

				step=1

				while not coord.should_stop():

					_ , C  , IS, LR , SL = sess.run([train_model.optimize,
													train_model.cost,
													train_model.global_step,
													train_model.learning_rate,
													train_model.slope])


					epoch_cost += C

					if (step % 50 == 0 or step==1):
						print("step[{:7d}] cost[{:2.5f}] lr[{:1.8f}] slope[{:1.8f}] time[{}]".format(step,C,LR,SL,time.time()-partial_time))
						partial_time=time.time()
						
					if (step%500 == 0):
						# save training parameters
						save_path = saver.save(sess,checkpoints_dir+'model_step'+str(step)+'.ckpt')
						print('Model saved!')

					if ((step % int(num_examples / config.batch_size) == 0) and (step is not 0)):

						# at each step we get the average cost over a batch, so divide by the 
						# number of batches in one epoch
						epoch_cost /=  (num_examples/config.batch_size)

						print('Completed epoch {:d} at step {:d} --> cost[{:.6f}]'.format(epoch_counter,step,epoch_cost))
						print('Epoch training time (seconds) = ',time.time()-EpochStartTime)

						# save summaries to tensorboard
						#summary = sess.run(merged_summary_op)
						#summary_writer.add_summary(summary, epoch_counter)

						# evaluation every "plot_every_epoch" epochs
						plot_every_epoch=1

						ValidationStartTime=time.time()

						# perform evaluation
						if((epoch_counter%plot_every_epoch)==0):

							layer_average_fw = np.zeros(config.num_layers)
							layer_average_bw = np.zeros(config.num_layers)

							z_logits_fw_hist={}
							z_logits_bw_hist={}

							accuracy , \
							binary_states_fw , binary_logits_fw, \
							binary_states_bw , binary_logits_bw = sess.run([val_model.accuracy,
																			val_model.binary_states_fw,
																			val_model.binary_logits_fw,
																			val_model.binary_states_bw,
																			val_model.binary_logits_bw])
								
								
							# get average activation of binary gates at each layer
							for layer in range(config.num_layers):

								layer_average_fw[layer] += np.mean(binary_states_fw['z_{:d}'.format(layer)])
								layer_average_bw[layer] += np.mean(binary_states_bw['z_{:d}'.format(layer)])


							# printout validation results
							print('Validation accuracy: {} '.format(accuracy))
							print('Validation time: {}'.format(time.time() - ValidationStartTime))
							
							trainingLogFile.write('{:d}\t{:.8f}\t{:.8f}\t{:.8f}\n'.format(epoch_counter,epoch_cost,LR,accuracy))
							trainingLogFile.flush()

							# Summary of binary states to TestLogFile

							outstring='{}-fw\t'.format(epoch_counter)

							#average number of boundaries per layer
							
							for i in range(config.num_layers):
								outstring+='{:.5f}\t'.format(layer_average_fw[i])

							outstring+='\n'

							testLogFile.write(outstring)

							outstring='{}-bw\t'.format(epoch_counter)
							#average number of boundaries per layer
							
							for i in range(config.num_layers):
								outstring+='{:.5f}\t'.format(layer_average_bw[i])

							outstring+='\n'
							outstring+='\n'

							testLogFile.write(outstring)
							testLogFile.flush()

							# Histograms of binary logits to file
							'''
							for i in range(config.num_layers):
								outfile=z_hist_dir+'forward_epoch{}_layer{}.npy'.format(0,i)
								np.save(outfile,z_logits_fw_hist[i],allow_pickle=False)

								outfile=z_hist_dir+'backward_epoch{}_layer{}.npy'.format(0,i)
								np.save(outfile,z_logits_bw_hist[i],allow_pickle=False)
							'''

							save_path = saver.save(sess,checkpoints_dir+'model_epoch'+str(epoch_counter)+'.ckpt')

						print('\n')	
						epoch_counter+=1
						epoch_cost=0.0
						EpochStartTime=time.time()

					step += 1

			except tf.errors.OutOfRangeError:
				print('---- Done Training: epoch limit reached ----')
			finally:
				coord.request_stop()

			coord.join(threads)

			save_path = saver.save(sess,checkpoints_dir+'model_end.ckpt')
			print("model saved in file: %s" % save_path)

	trainingLogFile.close()


def main(argv=None):  # pylint: disable=unused-argument
  train()

if __name__ == '__main__':
  tf.app.run()
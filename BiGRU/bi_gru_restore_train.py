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

import bi_rnn_model 


#################################
# Useful constant and paths
#################################
ExpNum = int(sys.argv[1])

num_examples=3696

class Configuration(object):

	logfile= open('TrainingExperiment'+str(ExpNum)+'.txt','r')
	lines = logfile.readlines()
	paramlines = [line for line in lines if line.startswith('##')]

	for params in paramlines:

		if 'learning rate' in params and 'update' not in params and 'decay' not in params:
			learning_rate = float (params.split(':',1)[1].strip() )
			print('learning_rate=',learning_rate)

		if 'train sequence length' in params:
			train_sequence_length = int ( params.split(':',1)[1].strip() )
			print('sequence_length=',train_sequence_length)
		
		if 'learning rate update step' in params:
			updating_step = int (params.split(':',1)[1].strip() )
			print('updating_step=',updating_step)

		if 'learning rate decay' in params:
			learning_decay = float (params.split(':',1)[1].strip() )
			print('learning_decay=',learning_decay)

		if 'dropout keep prob' in params:
			keep_prob = float (params.split(':',1)[1].strip() )
			print('keep_prob=',keep_prob)

		if 'batch size' in params:
			batch_size = int (params.split(':',1)[1].strip() )
			print('batch_size=',batch_size)

		if 'optimizer' in params:
			optimizer_choice= params.split(':',1)[1].strip()
			print('optimizer= ',optimizer_choice)

	last_epoch = int(lines[-1].split('\t')[0])
	print('last epoch =',last_epoch)


	audio_feat_dimension = 120
	audio_labels_dim=144

	num_epochs=5000
	
	n_hidden=217
	num_layers=5

checkpoints_dir='checkpoints/exp'+str(ExpNum)+'/'

tensorboard_dir='tensorboard/exp'+str(ExpNum)+'/'

trainingLogFile=open('TrainingExperiment'+str(ExpNum)+'.txt','a')


###################################
# Auxiliary functions
###################################

# Reads a single serialized SequenceExample
def read_my_file_format(filename_queue,feat_dimension=123):

    reader = tf.TFRecordReader()

    key, serialized_example = reader.read(filename_queue)

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(serialized_example,
                                               context_features={"length": tf.FixedLenFeature([],dtype=tf.int64)},
                                               sequence_features={"audio_feat":tf.FixedLenSequenceFeature([feat_dimension],dtype=tf.float32),
                                                                  "audio_labels":tf.FixedLenSequenceFeature([],dtype=tf.float32)}
                                        )

    return context_parsed['length'],sequence_parsed['audio_feat'],tf.to_int32(sequence_parsed['audio_labels'])

# training input pipeline
def input_pipeline(filenames,config):

    filename_queue = tf.train.string_input_producer(filenames, num_epochs=config.num_epochs, shuffle=True)
    
    sequence_length, audio_features, audio_labels = read_my_file_format(filename_queue,feat_dimension=config.audio_feat_dimension)

    audio_features_batch, audio_labels_batch , seq_length_batch = tf.train.batch([audio_features, audio_labels, sequence_length],
                                                    batch_size=config.batch_size,
                                                    num_threads=3,
                                                    capacity=5000,
                                                    dynamic_pad=True,
                                                    enqueue_many=False)

    return audio_features_batch, audio_labels_batch, seq_length_batch

# validation input pipeline
def input_pipeline_validation(filenames,batch_size,num_epochs=1):

    filename_queue = tf.train.string_input_producer(filenames, num_epochs=None, shuffle=False)
    
    sequence_length, audio_features, audio_labels = read_my_file_format(filename_queue)

    audio_features_batch, audio_labels_batch , seq_length_batch = tf.train.batch([audio_features, audio_labels, sequence_length],
                                                    batch_size=batch_size,
                                                    num_threads=3,
                                                    capacity=5000,
                                                    dynamic_pad=True,
                                                    enqueue_many=False)

    return audio_features_batch, audio_labels_batch, seq_length_batch


#################################
# Training module
#################################

def restore_train():

	config=Configuration()
	
	# list of input filenames + check existence
	filename_train=['/home/rtavaron/KaldiTimit/data/TRAIN/sequence_full_{:04d}.tfrecords'.format(i) \
					 for i in range(num_examples)]
	for f in filename_train:
		if not tf.gfile.Exists(f):
			raise ValueError('Failed to find file: ' + f)

	with open ('/home/rtavaron/KaldiTimit/data/CoreTestSentences_pickle2.dat','rb') as fp:
		testing_data = pickle.load(fp)

	print("Test data loaded")
	num_examples_val = len(testing_data)
	print("Number of test sentences : ",num_examples_val)


	# training graph		
	with tf.Graph().as_default():

		# extract batch examples
		with tf.device('/gpu:0'):
			with tf.name_scope('train_batch'):
				audio_features, audio_labels, seq_length = input_pipeline(filename_train,config)

		with tf.device('/gpu:0'):
			with tf.name_scope('validation_batch'):
				audio_features_val = tf.placeholder(dtype=tf.float32, shape=[None,None,config.audio_feat_dimension])
				audio_labels_val = tf.placeholder(dtype=tf.int32,shape=[None,None])
				seq_length_val = tf.placeholder(dtype=tf.int32,shape=[None])

		# audio features reconstruction
		with tf.device('/gpu:0'):
			with tf.variable_scope('model',reuse=None):
				print('Building training model:')
				train_model = bi_rnn_model.Model(audio_features,audio_labels,seq_length,config,is_training=True)
				print('done.\n')

		with tf.device('/gpu:0'):
			with tf.variable_scope('model',reuse=True):
				print('Building validation model:')
				val_model = bi_rnn_model.Model(audio_features_val,audio_labels_val,seq_length_val,config,is_training=None)
				print('done.')

		# variables initializer
		init_op = tf.local_variables_initializer()

		# save and restore all the variables.
		saver = tf.train.Saver(max_to_keep=None)
		
		# start session
		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
											  log_device_placement=False)) as sess:

			# tensorboard writer
			#train_writer = tf.summary.FileWriter(tensorboard_dir,sess.graph)

			# run initializer
			
			sess.run(init_op)
			print('Restoring variables...')
			saver.restore(sess,tf.train.latest_checkpoint(checkpoints_dir))
			print('variables loaded from ',tf.train.latest_checkpoint(checkpoints_dir))

			step=sess.run(train_model.global_step)
			print('last step = ',step)
			# start queue coordinator and runners
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(sess = sess, coord = coord)

			print('')
			print('## EXPERIMENT NUMBER ',ExpNum)
			print('## optimizer : ',config.optimizer_choice)
			print('## number of hidden layers : ',config.num_layers)
			print('## number of hidden units : ',config.n_hidden)
			print('## learning rate : ',config.learning_rate)
			print('## learning rate update steps: ',config.updating_step)
			print('## learning rate decay : ',config.learning_decay)
			print('## dropout keep prob (no dropout if 1.0) :',config.keep_prob)	
			print('## batch size : ',config.batch_size)		
			print('## number of steps: ',num_examples*config.num_epochs/config.batch_size)		
			print('## number of steps per epoch: ',num_examples/config.batch_size)
			print('')

			
			try:
				epoch_counter=config.last_epoch+1
				epoch_cost=0.0
				EpochStartTime=time.time()

				step=sess.run(train_model.global_step)

				while not coord.should_stop():

					_ , C  , IS, LR  = sess.run([train_model.optimize,
													train_model.cost,
													train_model.global_step,
													train_model.learning_rate])
					epoch_cost += C

					if (step % 50 == 0 or step==1):
						print("step[{:7d}] cost[{:2.5f}] lr[{:1.8f}]".format(step,C,LR))
					
					if (step %500==0):
						# save training parameters
						save_path = saver.save(sess,checkpoints_dir+'model_step'+str(step)+'.ckpt')
						print('Model saved!')
					
					if ((step % int(num_examples / config.batch_size) == 0) and (step is not 0)):

						# at each step we get the average cost over a batch, so divide by the 
						# number of batches in one epoch
						epoch_cost /=  (num_examples/config.batch_size)

						print('Completed epoch {:d} at step {:d} --> cost[{:.6f}]'.format(epoch_counter,step,epoch_cost))
						print('Epoch training time (seconds) = ',time.time()-EpochStartTime)
						
						#accuracy evaluation on each sentence
						#to avoid computing accuracy on padded frames
						out_every_epoch=1
							
						if((epoch_counter%out_every_epoch)==0):

							accuracy_list=[]
							num_frames_list=[]

							for i in range(num_examples_val):

								feat = np.array( testing_data[i][:,:-1])
								lab = np.array ( testing_data[i][:,-1])
								length = testing_data[i].shape[0]

								feat = np.expand_dims(feat,axis=0)
								lab = np.expand_dims(lab,axis=0)
								seq_length = np.expand_dims(length,axis=0)

								feed = {audio_features_val: feat,
								audio_labels_val: lab,
								seq_length_val: seq_length}

								# validation
								example_accuracy , labels , prediction = sess.run([val_model.accuracy,
																			val_model.labels,
																			val_model.prediction],feed_dict=feed)
								accuracy_list.append(example_accuracy)
								num_frames_list.append(length)

							accuracy_list=np.array(accuracy_list)
							num_frames_list=np.array(num_frames_list)

							accuracy = np.sum(accuracy_list*num_frames_list)/np.sum(num_frames_list)
							
							# printout validation results
							print('Validation accuracy : {} '.format(accuracy))
							
							trainingLogFile.write('{:d}\t{:.8f}\t{:.8f}\t{:.8f}\n'.format(epoch_counter,epoch_cost,LR,accuracy))
							trainingLogFile.flush()

							

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
  restore_train()

if __name__ == '__main__':
  tf.app.run()
from __future__ import print_function
from __future__ import division

# Avoid printing tensorflow log messages
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


import numpy as np
import time
import sys

import birnn_ctc_model 

#################################
# Useful constant and paths
#################################
ExpNum = int(sys.argv[1])

num_examples=3696

class Configuration(object):

	logfile= open('training_logs/TrainingExperiment'+str(ExpNum)+'.txt','r')
	lines = logfile.readlines()
	paramlines = [line for line in lines if line.startswith('##')]

	for params in paramlines:

		if 'learning rate' in params and 'update' not in params and 'decay' not in params:
			learning_rate = float (params.split(':',1)[1].strip() )
			print('learning_rate=',learning_rate)
		
		if 'learning rate update step' in params:
			updating_step = int (params.split(':',1)[1].strip() )
			print('updating_step=',updating_step)

		if 'learning rate decay' in params:
			learning_decay = float (params.split(':',1)[1].strip() )
			print('learning_decay=',learning_decay)

		if 'dropout keep prob' in params:
			keep_prob = float (params.split(':',1)[1].strip() )
			print('keep_prob=',keep_prob)

		if 'optimizer' in params:
			optimizer_choice= params.split(':',1)[1].strip()
			print('optimizer= ',optimizer_choice)

		if 'batch size' in params:
			batch_size=int(params.split(':',1)[1].strip())
			print('batch_size =', batch_size) 

	last_epoch = int(lines[-1].split('\t')[0])
	print('last epoch =',last_epoch)

	audio_feat_dimension = 123
	audio_labels_dim=61

	num_classes = audio_labels_dim + 1

	num_examples_val=192
	num_epochs=5000
	
	n_hidden=250
	num_layers=5


checkpoints_dir='checkpoints/exp'+str(ExpNum)+'/'

tensorboard_dir='tensorboard/exp'+str(ExpNum)+'/'

traininglog_dir='training_logs/'

trainingLogFile=open(traininglog_dir+'TrainingExperiment'+str(ExpNum)+'.txt','a')

###################################
# Auxiliary functions
###################################

#########################################
#indices mapping: from 61 to 39 phonemes
#########################################

timitPH = ['aa','ae','ah','ao','aw','ax','axh','axr','ay','b','bcl','ch',
			'd','dcl','dh','dx','eh','el','em','en','eng','epi','er','ey',
			'f','g','gcl','hC','hh','hv','ih','ix','iy','jh','k','kcl','l',
			'm','n','ng','nx','ow','oy','p','pau','pcl','q','r','s','sh',
			't','tcl','th','uh','uw','ux','v','w','y','z','zh']

timitAlloph = ['ao','axh','ax','axr','hv','ix','el','em','en','nx',
				'eng','zh','ux','pcl','tcl','kcl','bcl','dcl','pau','epi','q','gcl']

timitMainph = ['aa','ah','ah','er','hh','ih','l','m','n','n','ng',
				'sh','uw','hC','hC','hC','hC','hC','hC','hC','hC','hC']


indices_to_map_from = [i for i,val in enumerate(timitPH) if val in timitAlloph]
indices_to_map_to = [timitPH.index(ph) for ph in timitMainph]

mapping_count=0
mapping_dict={}

for i in range(len(timitPH)):
	if i not in indices_to_map_from:
		mapping_dict[i]=i
	else:
		mapping_dict[i]=indices_to_map_to[mapping_count]
		mapping_count+=1

#########################################
# Input pipelines
#########################################
# Reads a single serialized SequenceExample
def read_my_file_format(filename_queue,feat_dimension=123):

    reader = tf.TFRecordReader()

    key, serialized_example = reader.read(filename_queue)

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(serialized_example,
                                               context_features={"feat_length": tf.FixedLenFeature([],dtype=tf.int64),
                                               					"label_length": tf.FixedLenFeature([],dtype=tf.int64)},
                                               sequence_features={"audio_feat":tf.FixedLenSequenceFeature([feat_dimension],dtype=tf.float32),
                                                                  "audio_labels":tf.FixedLenSequenceFeature([],dtype=tf.float32)}
                                        )

    return tf.to_int32(context_parsed['feat_length']),tf.to_int32(context_parsed['label_length']),sequence_parsed['audio_feat'],tf.to_int32(sequence_parsed['audio_labels'])

# training input pipeline
def input_pipeline(filenames,config):

    filename_queue = tf.train.string_input_producer(filenames, num_epochs=config.num_epochs, shuffle=True)
    
    sequence_length, labels_length, audio_features, audio_labels = read_my_file_format(filename_queue,config.audio_feat_dimension)

    audio_features_batch, audio_labels_batch , seq_length_batch, labels_length_batch = tf.train.batch([audio_features, audio_labels, sequence_length,labels_length],
                                                    batch_size=config.batch_size,
                                                    num_threads=5,
                                                    capacity=5000,
                                                    dynamic_pad=True,
                                                    enqueue_many=False)

    return audio_features_batch, audio_labels_batch, seq_length_batch,labels_length_batch

# validation input pipeline
def input_pipeline_validation(filenames,batch_size,config,num_epochs=None):

    filename_queue = tf.train.string_input_producer(filenames, num_epochs=None, shuffle=False)
    
    sequence_length, labels_length, audio_features, audio_labels = read_my_file_format(filename_queue,config.audio_feat_dimension)

    audio_features_batch, audio_labels_batch , seq_length_batch, labels_length_batch= tf.train.batch([audio_features, audio_labels, sequence_length,labels_length],
                                                    batch_size=batch_size,
                                                    num_threads=5,
                                                    capacity=5000,
                                                    dynamic_pad=True,
                                                    enqueue_many=False)

    return audio_features_batch, audio_labels_batch, seq_length_batch, labels_length_batch

#################################
# Training module
#################################
def train():

	config=Configuration()

	# list of input filenames + check existence
	filename_train=['/home/rtavaron/CTC_BiLSTM/data/CTC_TRAIN/sequence_full_{:04d}.tfrecords'.format(i) \
					 for i in range(num_examples)]
	for f in filename_train:
		if not tf.gfile.Exists(f):
			raise ValueError('Failed to find file: ' + f)

	# list of input filenames + check existence
	filename_val=['/home/rtavaron/CTC_BiLSTM/data/CTC_CORETEST/sequence_full_{:04d}.tfrecords'.format(i) \
					 for i in range(config.num_examples_val)]
	for f in filename_val:
		if not tf.gfile.Exists(f):
			raise ValueError('Failed to find file: ' + f)


	# training graph		
	with tf.Graph().as_default():

		# extract batch examples
		with tf.device('/gpu:0'):
			with tf.name_scope('train_batch'):
				audio_features, audio_labels, seq_length ,lab_length = input_pipeline(filename_train,config)

			with tf.name_scope('validation_batch'):
				audio_features_val, audio_labels_val, seq_length_val , lab_length_val = \
				input_pipeline_validation(filename_val,batch_size=1,config=config)

		# audio features reconstruction
		with tf.device('/gpu:0'):
			with tf.variable_scope('model',reuse=None):
				print('Building training model:')
				train_model = birnn_ctc_model.Model(audio_features,audio_labels,seq_length,lab_length,config,is_training=True)
				print('done.\n')

		with tf.device('/gpu:0'):
			with tf.variable_scope('model',reuse=True):
				print('Building validation model:')
				val_model = birnn_ctc_model.Model(audio_features_val,audio_labels_val,seq_length_val,lab_length_val,config,is_training=None)
				print('done.')

		# variables initializer
		init_op = tf.local_variables_initializer()

		# save and restore all the variables.
		saver = tf.train.Saver(max_to_keep=10)

		# start session
		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
											  log_device_placement=False)) as sess:
	
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
					
					_ , C  , GS, LR = sess.run([train_model.optimize,
													train_model.cost,
													train_model.global_step,
													train_model.learning_rate])


					epoch_cost += C
						
					if (step %100 == 0):
						print("step[{:7d}] cost[{:2.5f}] lr[{:1.8f}]".format(step,C,LR))
					
					if (step%500==0):
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

						if((epoch_counter%1)==0):

							label_error_rate=0.0

							for i in range(config.num_examples_val):
								# validation
								decoded , sparse_labels  = sess.run([val_model.decoded,
																	val_model.sparse_labels])

								# mapping decoded labels
								dec_values=np.copy(decoded[0].values)

								for k,v in mapping_dict.iteritems():
									dec_values[decoded[0].values==k] = v

								indices=tf.constant(decoded[0].indices)
								values=tf.constant(dec_values)
								dense_shape=tf.constant(decoded[0].dense_shape)

								decoded = tf.SparseTensor(indices=indices,values=values,dense_shape=dense_shape)

								# mapping ground-truth labels
								true_values=np.copy(sparse_labels.values)

								for k,v in mapping_dict.iteritems():
									true_values[sparse_labels.values==k] = v						
								

								indices=tf.constant(sparse_labels.indices)
								values=tf.constant(true_values)
								dense_shape=tf.constant(sparse_labels.dense_shape)
								sparse_labels = tf.SparseTensor(indices=indices,values=values,dense_shape=dense_shape)
								
								ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded, tf.int32),sparse_labels))
								ler=ler.eval()
								label_error_rate += ler

								if i < 5:
									print('sentence {}.'.format(i))
									print('true labels:')
									print(true_values)
									print('decoded labels:')
									print(dec_values)
									print('LER = {}'.format(ler))

							label_error_rate/=config.num_examples_val
							# write summaries to tensorboard
							#train_writer.add_summary(summary, epoch_counter)

							# printout validation results
							print('')
							print('Validation LER: {} '.format(label_error_rate))
							
							trainingLogFile.write('{:d}\t{:.8f}\t{:.8f}\t{:.8f}\n'.format(epoch_counter,epoch_cost,LR,label_error_rate))
							trainingLogFile.flush()

							save_path = saver.save(sess,checkpoints_dir+'model_epoch'+str(epoch_counter)+'.ckpt')

						print('\n')	
						epoch_counter+=1
						epoch_cost=0.0
						epoch_ler=0.0
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
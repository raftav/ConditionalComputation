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

ExpNum=int(sys.argv[1])

num_examples_dev=400

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

		if 'number of hidden layers' in params:
			num_layers= int(params.split(':',1)[1].strip())
			print('num layers = ',num_layers)

		if 'number of hidden units' in params:
			n_hidden= int(params.split(':',1)[1].strip())
			print('num hidden units = ',n_hidden)


	audio_feat_dimension = 120
	audio_labels_dim=144

checkpoints_dir='checkpoints/exp'+str(ExpNum)+'/'
DevLogFile=open('DevExperiment'+str(ExpNum)+'.txt','a')

# Reads a single serialized SequenceExample
def read_my_file_format(filename_queue,feat_dimension=120):

    reader = tf.TFRecordReader()

    key, serialized_example = reader.read(filename_queue)

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(serialized_example,
                                               context_features={"length": tf.FixedLenFeature([],dtype=tf.int64)},
                                               sequence_features={"audio_feat":tf.FixedLenSequenceFeature([feat_dimension],dtype=tf.float32),
                                                                  "audio_labels":tf.FixedLenSequenceFeature([],dtype=tf.float32)}
                                        )

    return context_parsed['length'],sequence_parsed['audio_feat'],tf.to_int32(sequence_parsed['audio_labels'])

def input_pipeline_validation(filenames,batch_size):

    filename_queue = tf.train.string_input_producer(filenames, num_epochs=1, shuffle=False)
    
    sequence_length, audio_features, audio_labels = read_my_file_format(filename_queue)

    audio_features_batch, audio_labels_batch , seq_length_batch = tf.train.batch([audio_features, audio_labels, sequence_length],
                                                    batch_size=batch_size,
                                                    num_threads=3,
                                                    capacity=5000,
                                                    dynamic_pad=True,
                                                    enqueue_many=False)

    return audio_features_batch, audio_labels_batch, seq_length_batch

def validate():

	config=Configuration()
	
	# list of input filenames + check existence
	filename_dev=['/home/rtavaron/KaldiTimit/data/DEV/sequence_full_{:04d}.tfrecords'.format(i) \
					 for i in range(num_examples_dev)]
	for f in filename_dev:
		if not tf.gfile.Exists(f):
			raise ValueError('Failed to find file: ' + f)

	# training graph		
	with tf.Graph().as_default():

		with tf.device('/gpu:0'):
			with tf.name_scope('train_batch'):
				audio_features, audio_labels, seq_length = input_pipeline_validation(filename_dev,1)

		with tf.device('/gpu:0'):
			with tf.variable_scope('model'):
				print('Building validation model:')
				val_model = bi_rnn_model.Model(audio_features,audio_labels,seq_length,config,is_training=None)
				print('done.')

		# variables initializer
		init_op = tf.local_variables_initializer()

		# save and restore all the variables.
		saver = tf.train.Saver(max_to_keep=None)

		for restore_step in range(5000,21500,500):
			checkpoint_name=checkpoints_dir+'model_step'+str(restore_step)+'.ckpt'

			# start session
			with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
												  log_device_placement=False)) as sess:

				sess.run(init_op)
				print('Restoring variables...')
				saver.restore(sess,checkpoint_name)
				print('variables loaded from ',checkpoint_name)

				coord = tf.train.Coordinator()
				threads = tf.train.start_queue_runners(sess = sess, coord = coord)

				try:
					step=0
					accuracy_list=[]
					num_frames_list=[]
					
					while not coord.should_stop():
						example_accuracy , labels , prediction = sess.run([val_model.accuracy,
																				val_model.labels,
																				val_model.prediction])
						accuracy_list.append(example_accuracy)
						num_frames_list.append(labels.shape[1])
						print('step[{}] accuracy[{}]'.format(step,example_accuracy))
						step+=1

				except tf.errors.OutOfRangeError:
					accuracy_list=np.array(accuracy_list)
					num_frames_list=np.array(num_frames_list)

					accuracy = np.sum(accuracy_list*num_frames_list)/np.sum(num_frames_list)
					
					# printout validation results
					print('Training Step {}  --- Validation accuracy : {} '.format(restore_step,accuracy))
					DevLogFile.write('{}\t{}\n'.format(restore_step,accuracy))
					DevLogFile.flush()

				finally:
					coord.request_stop()

				coord.join(threads)

		DevLogFile.close()

def main(argv=None):  # pylint: disable=unused-argument
  validate()

if __name__ == '__main__':
  tf.app.run()
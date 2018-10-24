from __future__ import print_function
from __future__ import division

# Avoid printing tensorflow log messages
import os
import tensorflow as tf
from tensorflow.python import debug as tf_debug

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


import numpy as np
import time
import sys
import pickle

import bi_rnn_model

ExpNum=1
restore_epoch=24

class Configuration(object):

	audio_feat_dimension = 120

	audio_labels_dim=144
	
	n_hidden=250
	num_layers=5

checkpoints_dir='checkpoints/exp'+str(ExpNum)+'/'

test_file_dict={}

with open('/home/rtavaron/KaldiTimit/data/coretest_PickleFile_indeces.txt') as fp:
	for line in fp:
		value,key = line.split('\t')
		key=key.replace('\n','')
		value = value.replace('/home/local/IIT/lbadino/Data/KP/timit/test/','').replace('.npy','')
		test_file_dict[key]=value



posteriors_out_file=open('TimitTestLogPosteriors_Exp{}_epoch{}.txt'.format(ExpNum,restore_epoch),'w')

def write_posteriors(file_index,posteriors,num_total_files):

	print('Original file name = ',test_file_dict[str(file_index)])
	line_1 = test_file_dict[str(file_index)] + ' [\n'
	posteriors_out_file.write(line_1)

	for frame in range(posteriors.shape[1]):
		line=''
		for dim in range(posteriors.shape[2] - 1):
			line+='{:.10f} '.format( posteriors[0,frame,dim])

		if frame != (posteriors.shape[1] -1 ):
			line+='{:.10f}\n'.format(posteriors[0,frame,-1] )

		elif file_index != (num_total_files-1):
			line+='{:.10f} ]\n'.format(posteriors[0,frame,-1] )
		else:
			line+='{:.10f} ]'.format(posteriors[0,frame,-1] )
		posteriors_out_file.write(line)



def test():

	config=Configuration()

	with open ('/home/rtavaron/KaldiTimit/data/CoreTestSentences_pickle2.dat','rb') as fp:
		testing_data = pickle.load(fp)

	print("Test data loaded")
	num_examples_val = len(testing_data)
	print("Number of test sentences : ",num_examples_val)

	with tf.Graph().as_default():

		with tf.device('/cpu:0'):
			with tf.name_scope('validation_batch'):
				audio_features_val = tf.placeholder(dtype=tf.float32, shape=[None,None,config.audio_feat_dimension])
				audio_labels_val = tf.placeholder(dtype=tf.int32,shape=[None,None])
				seq_length_val = tf.placeholder(dtype=tf.int32,shape=[None])

		with tf.device('/cpu:0'):
			with tf.variable_scope('model'):
				val_model = bi_rnn_model.Model(audio_features_val,audio_labels_val,seq_length_val,config,is_training=None)

		init_op = tf.global_variables_initializer()
		saver = tf.train.Saver()

		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
											  log_device_placement=False)) as sess:

			sess.run(init_op)
			saver.restore(sess,checkpoints_dir+'model_step8500.ckpt')

			accuracy_list=[]
			num_frames_list=[]

			for i in range(num_examples_val):

				print('Evaluating sentence {}'.format(i))

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
				example_accuracy , labels , prediction , posteriors = sess.run([val_model.accuracy,
															val_model.labels,
															val_model.prediction,
															val_model.posteriors],feed_dict=feed)
				accuracy_list.append(example_accuracy)
				num_frames_list.append(length)

				print(posteriors.shape)

				#posteriors = np.divide(posteriors,priors)
				posteriors = np.log(posteriors)

				write_posteriors(i,posteriors,num_examples_val)





			accuracy_list=np.array(accuracy_list)
			num_frames_list=np.array(num_frames_list)

			accuracy = np.sum(accuracy_list*num_frames_list)/np.sum(num_frames_list)
			
			# printout validation results
			print('Validation accuracy : {} '.format(accuracy))
			posteriors_out_file.close()




def main(argv=None):  # pylint: disable=unused-argument
  test()

if __name__ == '__main__':
  tf.app.run()
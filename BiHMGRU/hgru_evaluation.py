from __future__ import print_function
from __future__ import division

# Avoid printing tensorflow log messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
import time
import sys
import math
import pickle

import hgru_model_evaluation


#################################
# Useful constant and paths
#################################
ExpNum = int(sys.argv[1])
restore_epoch=int(sys.argv[2])

print('Evaluation at epoch ',restore_epoch)

class Configuration(object):

	logfile= open('training_logs/TrainingExperiment'+str(ExpNum)+'.txt','r')
	lines = logfile.readlines()
	paramlines = [line for line in lines if line.startswith('##')]

	for params in paramlines:

		if 'learning rate' in params and 'update' not in params and 'decay' not in params:
			learning_rate = float (params.split(':',1)[1].strip() )
			print('learning_rate=',learning_rate)
		
		if 'slope_annealing_rate' in params:
			slope_annealing_rate = float (params.split(':',1)[1].strip() )
			print('slope_annealing_rate=',slope_annealing_rate)

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
			optimizer_choice = params.split(':',1)[1].strip()
			print('optimizer =',optimizer_choice)

		if 'batch size' in params:
			batch_size=int(params.split(':',1)[1].strip())
			print('batch size =',batch_size)

		if 'number of hidden layers' in params:
			num_layers=int(params.split(':',1)[1].strip())
			print('n hidden layers=',num_layers)

		if 'number of hidden units' in params:
			n_hidden=int(params.split(':',1)[1].strip())
			print('n hidden =',n_hidden)

	audio_feat_dimension = 120

	num_classes = 144

	num_examples_val=192


checkpoints_dir='checkpoints/exp'+str(ExpNum)+'/'
z_states_dir='z_states/exp'+str(ExpNum)+'/'

traininglog_dir='training_logs/'


#################################
# Training module
#################################

def eval():

	config=Configuration()

	# list of input filenames + check existence
	with open ('/home/rtavaron/KaldiTimit/data/CoreTestSentences_pickle2.dat','rb') as fp:
		testing_data = pickle.load(fp)

	print("Test data loaded")
	num_examples_val = len(testing_data)
	print("Number of test sentences : ",num_examples_val)
			

	with tf.Graph().as_default():

		# extract batch examples
		with tf.device('/gpu:0'):

			with tf.name_scope('validation_batch'):
				audio_features_val = tf.placeholder(dtype=tf.float32, shape=[None,None,config.audio_feat_dimension])
				audio_labels_val = tf.placeholder(dtype=tf.int32,shape=[None,None])
				seq_length_val = tf.placeholder(dtype=tf.int32,shape=[None])

		#with tf.device('/cpu:0'):
		with tf.variable_scope('model'):
			print('Building validation model:')
			start=time.time()
			val_model = hgru_model_evaluation.Model(audio_features_val,audio_labels_val,seq_length_val,config,is_training=None)
			print('Graph creation time = ',time.time() - start)
			print('done.')

		# variables initializer
		print('Initializer creation')
		start=time.time()
		init_op = tf.local_variables_initializer()
		print('creation time = ',time.time() - start)
		print('Done')
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
			saver.restore(sess,checkpoints_dir+'model_epoch'+str(restore_epoch)+'.ckpt')
			print('variables loaded from ',tf.train.latest_checkpoint(checkpoints_dir))

			print('')
			print('## EXPERIMENT NUMBER ',ExpNum)
			print('## binary units : deterministic')	
			print('## optimizer : ',config.optimizer_choice)
			print('## number of hidden layers : ',config.num_layers)
			print('## number of hidden units : ',config.n_hidden)
			print('## learning rate : ',config.learning_rate)
			print('## learning rate update steps: ',config.updating_step)
			print('## learning rate decay : ',config.learning_decay)
			print('## slope_annealing_rate : ',config.slope_annealing_rate)	
			print('## dropout keep prob (no dropout if 1.0) :',config.keep_prob)	
			print('## batch size : ',config.batch_size)		
			print('')


			accuracy=0.0
			layer_average_fw = np.zeros(config.num_layers)
			layer_average_bw = np.zeros(config.num_layers)

			for i in range(config.num_examples_val):
				feat = np.array( testing_data[i][:,:-1])
				lab = np.array ( testing_data[i][:,-1])
				length = testing_data[i].shape[0]

				feat = np.expand_dims(feat,axis=0)
				lab = np.expand_dims(lab,axis=0)
				length = np.expand_dims(length,axis=0)

				feed = {audio_features_val: feat,
						audio_labels_val: lab,
						seq_length_val: length}

				sentence_accuracy, true_labels, prediction , \
				binary_states_fw , binary_logits_fw, activations_norm_fw, \
				binary_states_bw , binary_logits_bw, activations_norm_bw = sess.run([val_model.accuracy,
																			val_model.labels,
																			val_model.prediction,
																			val_model.binary_states_fw,
																			val_model.binary_logits_fw,
																			val_model.activations_norm_fw,
																			val_model.binary_states_bw,
																			val_model.binary_logits_bw,
																			val_model.activations_norm_bw],feed_dict=feed)

				

				print('Sentence {}'.format(i))
				print('true labels:')
				print(true_labels)
				print('predicted labels:')
				print(prediction)
				print('Accuravy = {}'.format(sentence_accuracy))

				# true_labels and predicion shape is (1,maxtime)
				# I squeeze them into (maxtime)
				true_labels = np.squeeze(true_labels,axis=0)
				predicion = np.squeeze(prediction,axis=0)

				true_labels_file=open(z_states_dir+'labels_epoch{:d}_sentence{:d}.npy'.format(restore_epoch,i),'w')
				predictions_file=open(z_states_dir+'predictions_epoch{:d}_sentence{:d}.npy'.format(restore_epoch,i),'w')

				z_states_file_fw=open(z_states_dir+'states_forward_epoch{:d}_sentence{:d}.npy'.format(restore_epoch,i),'w')
				activation_file_fw=open(z_states_dir+'activations_forward_norm_epoch{:d}_sentence{:d}.npy'.format(restore_epoch,i),'w')
				z_logits_file_fw=open(z_states_dir+'logit_states_forward_epoch{:d}_sentence{:d}.npy'.format(restore_epoch,i),'w')

				z_states_file_bw=open(z_states_dir+'states_backward_epoch{:d}_sentence{:d}.npy'.format(restore_epoch,i),'w')
				activation_file_bw=open(z_states_dir+'activations_backward_norm_epoch{:d}_sentence{:d}.npy'.format(restore_epoch,i),'w')
				z_logits_file_bw=open(z_states_dir+'logit_states_backward_epoch{:d}_sentence{:d}.npy'.format(restore_epoch,i),'w')

				print('')
				print('Number of acoustic frames = ',feat.shape[1])
				print('Number of forward detected boundaries layer 0 = ',np.sum(binary_states_fw['z_0']))
				print('Number of backward detected boundaries layer 0 = ',np.sum(binary_states_bw['z_0']))

				# binary_states[z_0] original shape is (1,maxtime,1)
				# we squeeze it inot (1,maxtime)
				#print('binary states shape:')
				#print(binary_states_fw['z_0'].shape)

				binary_states_fw['z_0'] = np.squeeze(binary_states_fw['z_0'],axis=2)
				binary_logits_fw['z_tilda_0'] = np.squeeze(binary_logits_fw['z_tilda_0'],axis=2)

				binary_states_bw['z_0'] = np.squeeze(binary_states_bw['z_0'],axis=2)
				binary_logits_bw['z_tilda_0'] = np.squeeze(binary_logits_bw['z_tilda_0'],axis=2)

				# gating_weights[0] original shape is (maxtime,1)
				# we squeeze it into (maxtime)
				# and we then expand it into (1,maxtime) to make it equal to binary_states[z_0].shape
				
				all_activation_fw = activations_norm_fw['0']
				all_states_fw = binary_states_fw['z_0']
				all_logits_fw = binary_logits_fw['z_tilda_0']
				
				all_activation_bw = activations_norm_bw['0']
				all_states_bw = binary_states_bw['z_0']
				all_logits_bw = binary_logits_bw['z_tilda_0']
				
				for layer in range(1,config.num_layers):

					layer_average_fw[layer] += np.mean(binary_states_fw['z_{:d}'.format(layer)])
					print('Number of forward detected boundaries layer {} = {}'.format(layer,np.sum(binary_states_fw['z_{}'.format(layer)])))

					binary_states_fw['z_{:d}'.format(layer)] = np.squeeze(binary_states_fw['z_{:d}'.format(layer)],axis=2)
					binary_logits_fw['z_tilda_{:d}'.format(layer)] = np.squeeze(binary_logits_fw['z_tilda_{:d}'.format(layer)],axis=2)

					
					all_states_fw = np.concatenate((all_states_fw,binary_states_fw['z_{:d}'.format(layer)]),axis=0)
					all_activation_fw = np.concatenate((all_activation_fw,activations_norm_fw['{}'.format(layer)]),axis=0)
					all_logits_fw = np.concatenate((all_logits_fw,binary_logits_fw['z_tilda_{:d}'.format(layer)]),axis=0)
					
					layer_average_bw[layer] += np.mean(binary_states_bw['z_{:d}'.format(layer)])
					print('Number of backward detected boundaries layer {} = {}'.format(layer,np.sum(binary_states_bw['z_{}'.format(layer)])))

					binary_states_bw['z_{:d}'.format(layer)] = np.squeeze(binary_states_bw['z_{:d}'.format(layer)],axis=2)
					binary_logits_bw['z_tilda_{:d}'.format(layer)] = np.squeeze(binary_logits_bw['z_tilda_{:d}'.format(layer)],axis=2)

					
					all_states_bw = np.concatenate((all_states_bw,binary_states_bw['z_{:d}'.format(layer)]),axis=0)
					all_activation_bw = np.concatenate((all_activation_bw,activations_norm_bw['{}'.format(layer)]),axis=0)
					all_logits_bw = np.concatenate((all_logits_bw,binary_logits_bw['z_tilda_{:d}'.format(layer)]),axis=0)
					
				accuracy += sentence_accuracy

				print('all_states shape = ',all_states_fw.shape)
				print('all activations norm shape ',all_activation_fw.shape)
				print('all logits shape = ',all_logits_fw.shape)
				
				np.save(true_labels_file,true_labels,allow_pickle=False)
				np.save(predictions_file,prediction,allow_pickle=False)

				np.save(z_states_file_fw,all_states_fw,allow_pickle=False)
				np.save(activation_file_fw,all_activation_fw,allow_pickle=False)
				np.save(z_logits_file_fw,all_logits_fw,allow_pickle=False)

				np.save(z_states_file_bw,all_states_bw,allow_pickle=False)
				np.save(activation_file_bw,all_activation_bw,allow_pickle=False)
				np.save(z_logits_file_bw,all_logits_bw,allow_pickle=False)

				print('')
				print('')
				
			accuracy /= config.num_examples_val
			print('Overall accuracy = ',accuracy)



def main(argv=None):  # pylint: disable=unused-argument
  eval()

if __name__ == '__main__':
  tf.app.run()
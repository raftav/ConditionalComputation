from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from itertools import groupby
import sys
import os

exp_num=1
epoch=65

#outfile='z_states/computational_saving_exp{}_{}.txt'.format(exp_num,epoch)
#fp=open(outfile,'w')

num_test_sentences=192
num_hidden_layers=5


copies_percentage_forward={}
copies_percentage_backward={}

for i in range(num_hidden_layers):
	copies_percentage_forward[i+1]=0.0
	copies_percentage_backward[i+1]=0.0

for sentence in range(num_test_sentences):

	states_bw=np.load('z_states/exp{}/states_backward_epoch{}_sentence{}.npy'.format(exp_num,epoch,sentence))
	states_fw=np.load('z_states/exp{}/states_forward_epoch{}_sentence{}.npy'.format(exp_num,epoch,sentence))

	sentence_length = states_fw[i].shape[0]
	print('Sentece {}:'.format(sentence))

	not_boundary_fw={}
	not_boundary_bw={}

	for i in range(num_hidden_layers):
		not_boundary_fw[i] = np.where(states_fw[i]==0.0)
		not_boundary_bw[i] = np.where(states_bw[i]==0.0)

		print('Forward layer {} not boundaries = {} over {}'.format(i+1,not_boundary_fw[i][0].shape[0],sentence_length))
		print('Backward layer {} not boundaries = {} over {}'.format(i+1,not_boundary_bw[i][0].shape[0],sentence_length))
	print('')

	for i in range(num_hidden_layers-1):
		num_copies_fw = np.intersect1d(not_boundary_fw[i][0],not_boundary_fw[i+1][0])
		num_copies_fw = len(num_copies_fw)

		percentage_copies_fw = num_copies_fw / sentence_length

		print('Forward layer {} number of copies = {} over {}'.format(i+1,num_copies_fw,sentence_length))
		print('Forward layer {} percentage copies = {}'.format(i+1,percentage_copies_fw))

		num_copies_bw = np.intersect1d(not_boundary_bw[i][0],not_boundary_bw[i+1][0])
		num_copies_bw = len(num_copies_bw)

		percentage_copies_bw = num_copies_bw / sentence_length

		print('Backward layer {} number of copies = {} over {}'.format(i+1,num_copies_bw,sentence_length))
		print('Backward layer {} percentage copies = {}'.format(i+1,percentage_copies_bw))
		
		copies_percentage_forward[i+2] += percentage_copies_fw
		copies_percentage_backward[i+2] += percentage_copies_bw
		print('')

	print('')
	print('')

for i in range(num_hidden_layers):
	copies_percentage_forward[i+1]/=num_test_sentences
	copies_percentage_backward[i+1]/=num_test_sentences

	print('Forward layer {} dataset-level percentage of copies = {}'.format(i+1,copies_percentage_forward[i+1]))
	print('Backward layer {} dataset-level percentage of copies = {}'.format(i+1,copies_percentage_backward[i+1]))
	print('Layer {} average = {}'.format(i+1,(copies_percentage_forward[i+1] + copies_percentage_backward[i+1])/2.0))
	print('')
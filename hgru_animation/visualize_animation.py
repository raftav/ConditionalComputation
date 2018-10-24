from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from itertools import groupby
import sys
import os
import wave

import matplotlib.animation as animation

##########################
# Parameters
##########################

exp_num=1
epoch=74

sentence=129


num_hidden_layers=5


##########################
# Data Loading
##########################
labels=np.load('data/labels_epoch{}_sentence{}.npy'.format(epoch,sentence))
sequence_length=labels.shape[0]
print(sequence_length)

predictions=np.load('data/predictions_epoch{}_sentence{}.npy'.format(epoch,sentence))
predictions=predictions[0]


states=np.load('data/states_forward_epoch{}_sentence{}.npy'.format(epoch,sentence))
activations=np.load('data/activations_forward_norm_epoch{}_sentence{}.npy'.format(epoch,sentence))
logits=np.load('data/logit_states_forward_epoch{}_sentence{}.npy'.format(epoch,sentence))

gates=np.ones_like(states)
activations=np.round(activations,4)
print('Data loaded')

#####################################
# Colormap for activation functions
#####################################
binary_color=['y','k']

def get_colors(inp, colormap, vmin=None, vmax=None):
    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(inp))

######################################
# Plotting code
######################################

f,ax = plt.subplots(2,figsize=(15,10))

####################################
# True Labels
####################################

lab_min_val=0
lab_max_val=183



####################################
# Predictions
####################################
ax[0].set_yticklabels([])
ax[0].set_yticks([])

y_labels_position=[0.5, 1.5, 2.3, 3.0, 3.8, 4.5, 5.3, 6.0, 6.8, 7.5, 8.3, 9.0]

y_labels=["labels",
		r"$ \vert \vert \mathbf{h}^{1} \vert \vert $",
		r"$z^{1}$",
		r"$ \vert \vert \mathbf{h}^{2} \vert \vert $",
		r"$z^{2}$",
		r"$ \vert \vert \mathbf{h}^{3} \vert \vert $",
		r"$z^{3}$",
		r"$ \vert \vert \mathbf{h}^{4} \vert \vert $",
		r"$z^{4}$",
		r"$ \vert \vert \mathbf{h}^{5} \vert \vert $",
		r"$z^{5}$",
		"prediction"]

ax[0].set_yticks(y_labels_position)
ax[0].set_yticklabels(y_labels)

ax[0].set_xlim(0.0,float(sequence_length))
ax[0].set_ylim(0.0,9.5)

ax[0].set_xlabel('timestep')


ax[0].barh(y=0.00,
		height=0.8,
		left=float(0),
		width=1.0,
		align='edge',
		color=get_colors(labels[0],plt.cm.Set1,lab_min_val,lab_max_val),
		edgecolor='w')

y=1.0

for layer in range(0,5):
	max_val=np.amax(activations[layer][0:sequence_length])
	min_val=np.amin(activations[layer][0:sequence_length])

	#hidden layer activation
	ax[0].barh(y=y,
			height=0.9,
			left=float(0),
			width=1.0,
			align='edge',
			color=get_colors(activations[layer][0],plt.cm.jet,min_val,max_val),
			edgecolor='w')

	y+=1.0
	h=0.5
	# hidden layer binary states
	ax[0].barh(y=y,
			height=0.4,
			left=float(0),
			width=1.0,
			align='edge',
			color=binary_color[int(states[0][0])],
			edgecolor='w')
	y+=0.5

ax[0].barh(y=8.5,
	height=0.9,
	left=float(0),
	width=1.0,
	align='edge',
	color=get_colors(predictions[0],plt.cm.Set1,lab_min_val,lab_max_val),
	edgecolor='w')



##########################################


spf = wave.open('SX126.wav','r')
signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')

x=np.linspace(0,sequence_length,len(signal))


####################################

def generate_frame(t=0):
	while t<sequence_length:
		t+=1
		yield t


def run_frame(index):


	ax[0].barh(y=0.00,
			height=0.8,
			left=float(index),
			width=1.0,
			align='edge',
			color=get_colors(labels[index],plt.cm.Set1,lab_min_val,lab_max_val),
			edgecolor='w')

	y=1.0

	for layer in range(0,5):
		max_val=np.amax(activations[layer][0:sequence_length])
		min_val=np.amin(activations[layer][0:sequence_length])

		#hidden layer activation

		ax[0].barh(y=y,
				height=0.9,
				left=float(index),
				width=1.0,
				align='edge',
				color=get_colors(activations[layer][index],plt.cm.jet,min_val,max_val),
				edgecolor='w')

		y+=1.0
		h=0.5
		# hidden layer binary states

		ax[0].barh(y=y,
				height=0.4,
				left=float(index),
				width=1.0,
				align='edge',
				color=binary_color[int(states[layer][index])],
				edgecolor='w')
		y+=0.5

	ax[0].barh(y=8.5,
		height=0.9,
		left=float(index),
		width=1.0,
		align='edge',
		color=get_colors(predictions[index],plt.cm.Set1,lab_min_val,lab_max_val),
		edgecolor='w')

	ax[1].clear()
	ax[1].plot(x,signal)
	ax[1].set_xlim(0.0,float(sequence_length))
	ax[1].axvline(index,linewidth=2.0,color='r')




plt.tight_layout()
ani = animation.FuncAnimation(f, run_frame,frames=sequence_length,interval=1000,
                              repeat=True)


#plt.show()

Writer = animation.writers['ffmpeg']

fps=30
writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=2800)
ani.save('animation_sentence{}_fps{}.mp4'.format(sentence,fps),writer=writer)

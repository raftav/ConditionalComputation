from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from itertools import groupby
import sys
import os

##########################
# Parameters
##########################

exp_num=1
epoch=74

sentence=129
sequence_length=246

num_hidden_layers=5


##########################
# Data Loading
##########################
labels=np.load('data/labels_epoch{}_sentence{}.npy'.format(epoch,sentence))


print(labels.shape)

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

f,ax = plt.subplots(figsize=(15,5))

####################################
# True Labels
####################################
lab_min_val=0
lab_max_val=183

for i in range(sequence_length):
	ax.barh(y=0.00,
			height=0.8,
			left=float(i),
			width=1.0,
			align='edge',
			color=get_colors(labels[i],plt.cm.Set1,lab_min_val,lab_max_val),
			edgecolor='w')

####################################
# Predictions
####################################
for i in range(sequence_length):
	ax.barh(y=8.5,
			height=0.9,
			left=float(i),
			width=1.0,
			align='edge',
			color=get_colors(predictions[i],plt.cm.Set1,lab_min_val,lab_max_val),
			edgecolor='w')


y_labels_position=[0.5]


####################################








y=1.0

for layer in range(0,5):
	max_val=np.amax(activations[layer][0:sequence_length])
	min_val=np.amin(activations[layer][0:sequence_length])

	#hidden layer activation
	for i in range(sequence_length):
		ax.barh(y=y,
				height=0.9,
				left=float(i),
				width=1.0,
				align='edge',
				color=get_colors(activations[layer][i],plt.cm.jet,min_val,max_val),
				edgecolor='w')

	y_labels_position.append(y+0.5)

	y+=1.0
	h=0.5
	# hidden layer binary states
	for i in range(sequence_length):
		ax.barh(y=y,
				height=0.4,
				left=float(i),
				width=1.0,
				align='edge',
				color=binary_color[int(states[0][i])],
				edgecolor='w')

	y_labels_position.append(y+0.3)
	y+=0.5


print('last y = ',y)

ax.set_yticklabels([])
ax.set_yticks([])

y_labels_position.append(9.0)

print(y_labels_position)
#y_labels_position=[0.5,1.5,2.3,4.0,4.8,6.5,7.3,9.0,9.8,11.5,12.3,14.0]

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

ax.set_yticks(y_labels_position)
ax.set_yticklabels(y_labels)

ax.set_xlim(0.0,float(sequence_length))
ax.set_ylim(0.0,9.5)

ax.set_xlabel('timestep')

plt.show()
'''
plt.tight_layout()
plotname='activation_forward_exp{}_epoch{}_sentence{}'.format(exp_num,epoch,sentence)
plt.savefig(plotname+'.png',dpi=300)

plt.clf()
'''
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
epoch=65

sentence=191
sequence_length=250

num_hidden_layers=5


##########################
# Data Loading
##########################
labels=np.load('z_states/exp{}/labels_epoch{}_sentence{}.npy'.format(exp_num,epoch,sentence))
predictions=np.load('z_states/exp{}/predictions_epoch{}_sentence{}.npy'.format(exp_num,epoch,sentence))
predictions=predictions[0]


states=np.load('z_states/exp{}/states_forward_epoch{}_sentence{}.npy'.format(exp_num,epoch,sentence))
activations=np.load('z_states/exp{}/activations_forward_norm_epoch{}_sentence{}.npy'.format(exp_num,epoch,sentence))
logits=np.load('z_states/exp{}/logit_states_forward_epoch{}_sentence{}.npy'.format(exp_num,epoch,sentence))

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

# the following portion of code sets the layout for the horizontal bars
# also fixes the axes min and max values.
# the y-axis will have min=0.0 and max= float(num_hidden_layers)*2.0+2.0
# the x-axis will have min=0.0 and max=sequence_length

#bottom=np.arange(0.0,float(num_hidden_layers)*2.0+2.0,1.0)
'''
bottom=[0.0]
height=[0.8]

#true labels
bottom.append(1.0)
height.append(0.8)
# hidden layers
val=2.0
for i in range(num_hidden_layers):
	
	bottom.append(val)
	height.append(0.8)
	bottom.append(val+1.0)
	height.append(0.8)
	bottom.append(val+1.5)
	height.append(0.4)
	val+=2.5

# predictions
bottom.append(1.5+2.5*float(num_hidden_layers))
height.append(0.8)

bottom = np.asarray(bottom)
print(len(bottom))

height = np.asarray(height)
print(len(height))

width=np.full(bottom.shape,float(sequence_length))

print(bottom)
print(height)

ax.barh(bottom=bottom,
		height=height,
		left=0.0,
		width=width,
		align='edge',
		color='none',
		linewidth=1.0,edgecolor='k')
'''
# because of the first barh call, now setting a color brick correnspondent
# to layer l and timestep i is easy, because timestep i starts at x=i and has length 1.0.
# For example, to place a black brick at layer 1 and time step 20 do the following.
#
#       ax.barh(bottom=1.00,
#				height=0.8,
#				left=20.0,
#				width=1.0,
#				align='edge',
#				color='k',
#				edgecolor='none')

#width=np.full(bottom.shape,float(sequence_length))

####################################
# True Labels
####################################
lab_min_val=0
lab_max_val=183

for i in range(sequence_length):
	ax.barh(bottom=0.00,
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
	ax.barh(bottom=13.50,
			height=0.8,
			left=float(i),
			width=1.0,
			align='edge',
			color=get_colors(predictions[i],plt.cm.Set1,lab_min_val,lab_max_val),
			edgecolor='w')

####################################
# hidden layer 1 
####################################

# activation norm
max_val=np.amax(activations[0][0:sequence_length])
min_val=np.amin(activations[0][0:sequence_length])
#max_val=500.0
#min_val=0.0

print('max val = ',max_val)
print('min val = ',min_val)

for i in range(sequence_length):
	ax.barh(bottom=1.00,
			height=0.9,
			left=float(i),
			width=1.0,
			align='edge',
			color=get_colors(activations[0][i],plt.cm.jet,min_val,max_val),
			edgecolor='w')

# hidden layer 1 binary states

for i in range(sequence_length):
	ax.barh(bottom=2.00,
			height=0.9,
			left=float(i),
			width=1.0,
			align='edge',
			color=binary_color[int(states[0][i])],
			edgecolor='w')	

# hidden layer 1 gates
for i in range(sequence_length):
	ax.barh(bottom=3.00,
			height=0.3,
			left=float(i),
			width=1.0,
			align='edge',
			color=get_colors(gates[0][i],plt.cm.binary,0.0,1.0),
			edgecolor='k')


####################################
# hidden layer 2 
####################################

# activation norm
max_val=np.amax(activations[1][0:sequence_length])
min_val=np.amin(activations[1][0:sequence_length])

for i in range(sequence_length):
	ax.barh(bottom=3.50,
			height=0.9,
			left=float(i),
			width=1.0,
			align='edge',
			color=get_colors(activations[1][i],plt.cm.jet,min_val,max_val),
			edgecolor='w')

# hidden layer 2 binary states
for i in range(sequence_length):
	ax.barh(bottom=4.50,
			height=0.9,
			left=float(i),
			width=1.0,
			align='edge',
			color=binary_color[int(states[1][i])],
			edgecolor='w')	

# hidden layer 2 gates
for i in range(sequence_length):
	ax.barh(bottom=5.50,
			height=0.3,
			left=float(i),
			width=1.0,
			align='edge',
			color=get_colors(gates[1][i],plt.cm.binary,0.0,1.0),
			edgecolor='k')

####################################
# hidden layer 3 
####################################

# activation norm
max_val=np.amax(activations[2][0:sequence_length])
min_val=np.amin(activations[2][0:sequence_length])

for i in range(sequence_length):
	ax.barh(bottom=6.00,
			height=0.9,
			left=float(i),
			width=1.0,
			align='edge',
			color=get_colors(activations[2][i],plt.cm.jet,min_val,max_val),
			edgecolor='w')

# hidden layer 2 binary states
for i in range(sequence_length):
	ax.barh(bottom=7.00,
			height=0.9,
			left=float(i),
			width=1.0,
			align='edge',
			color=binary_color[int(states[2][i])],
			edgecolor='w')	

# hidden layer 3 gates
for i in range(sequence_length):
	ax.barh(bottom=8.00,
			height=0.3,
			left=float(i),
			width=1.0,
			align='edge',
			color=get_colors(gates[2][i],plt.cm.binary,0.0,1.0),
			edgecolor='k')


####################################
# hidden layer 4
####################################

# activation norm
max_val=np.amax(activations[3][0:sequence_length])
min_val=np.amin(activations[3][0:sequence_length])

for i in range(sequence_length):
	ax.barh(bottom=8.50,
			height=0.9,
			left=float(i),
			width=1.0,
			align='edge',
			color=get_colors(activations[3][i],plt.cm.jet,min_val,max_val),
			edgecolor='w')

# hidden layer 2 binary states
for i in range(sequence_length):
	ax.barh(bottom=9.50,
			height=0.9,
			left=float(i),
			width=1.0,
			align='edge',
			color=binary_color[int(states[3][i])],
			edgecolor='w')	

# hidden layer 4 gates
for i in range(sequence_length):
	ax.barh(bottom=10.50,
			height=0.3,
			left=float(i),
			width=1.0,
			align='edge',
			color=get_colors(gates[3][i],plt.cm.binary,0.0,1.0),
			edgecolor='k')


####################################
# hidden layer 5
####################################

# activation norm
max_val=np.amax(activations[4][0:sequence_length])
min_val=np.amin(activations[4][0:sequence_length])

for i in range(sequence_length):
	ax.barh(bottom=11.00,
			height=0.9,
			left=float(i),
			width=1.0,
			align='edge',
			color=get_colors(activations[4][i],plt.cm.jet,min_val,max_val),
			edgecolor='w')

# hidden layer 2 binary states
for i in range(sequence_length):
	ax.barh(bottom=12.00,
			height=0.9,
			left=float(i),
			width=1.0,
			align='edge',
			color=binary_color[int(states[4][i])],
			edgecolor='w')	

# hidden layer 5 gates
for i in range(sequence_length):
	ax.barh(bottom=13.0,
			height=0.3,
			left=float(i),
			width=1.0,
			align='edge',
			color=get_colors(gates[4][i],plt.cm.binary,0.0,1.0),
			edgecolor='k')



ax.set_yticklabels([])
ax.set_yticks([])

y_labels_position=[0.5,1.5,2.3,3.0,4.0,4.8,5.5,6.5,7.3,8.0,9.0,9.8,10.5,11.5,12.3,13.0,14.0]

y_labels=["labels",
		r"$ \vert \vert \mathbf{h}^{1} \vert \vert $",
		r"$z^{1}$",
		r"$g^{1}$",
		r"$ \vert \vert \mathbf{h}^{2} \vert \vert $",
		r"$z^{2}$",
		r"$g^{2}$",
		r"$ \vert \vert \mathbf{h}^{3} \vert \vert $",
		r"$z^{3}$",
		r"$g^{3}$",
		r"$ \vert \vert \mathbf{h}^{4} \vert \vert $",
		r"$z^{4}$",
		r"$g^{4}$",
		r"$ \vert \vert \mathbf{h}^{5} \vert \vert $",
		r"$z^{5}$",
		r"$g^{5}$",
		"prediction"]

ax.set_yticks(y_labels_position)
ax.set_yticklabels(y_labels)

'''
scalarmappaple = plt.cm.ScalarMappable(norm=plt.Normalize(min_val, max_val), cmap=plt.cm.jet)
nValues=np.arange(min_val,max_val)
scalarmappaple.set_array(nValues)
plt.colorbar(scalarmappaple,orientation='horizontal',shrink=0.3)

scalarmappaple = plt.cm.ScalarMappable(norm=plt.Normalize(lab_min_val, lab_max_val), cmap=plt.cm.Set1)
nValues=np.arange(lab_min_val,lab_max_val)
scalarmappaple.set_array(nValues)
plt.colorbar(scalarmappaple,orientation='horizontal',shrink=0.3)
'''

ax.set_xlim(0.0,float(sequence_length))
ax.set_ylim(0.0,14.5)

ax.set_xlabel('timestep')

plt.tight_layout()
plotname='z_states/activation_forward_exp{}_epoch{}_sentence{}'.format(exp_num,epoch,sentence)
plt.savefig(plotname+'.png',dpi=300)
#plt.savefig(plotname+'.svg',fmt='svg')

plt.clf()
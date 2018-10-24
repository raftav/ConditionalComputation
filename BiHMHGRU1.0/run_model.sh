#!/bin/bash

# EXPERIMENT NUMBER
exp_num=1

# learning rate
lr=0.0001

# slope annealing rate
sar=0.005

# update step
us=184

# learning rate deacy
lrd=1.0

# dropout keep prob
kp=1.0

# batch size
bs=1

#optimizer
opt=adam

# lambda l2
l2=0.0

#########################
#########################

source activate tensorflow1.2_py2.7

CUDA_VISIBLE_DEVICES=4 python -u hgru_train_multireader.py $exp_num $lr $sar $us $lrd $kp $bs $opt $l2 >> 'out_exp'$exp_num'.txt' 2>'out_exp'$exp_num'.err' &

source deactivate
disown



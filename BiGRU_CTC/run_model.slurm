#!/bin/bash

#SBATCH --job-name=birnn-ctc
#SBATCH --error=tensorflow-test-%j.err
#SBATCH --output=tensorflow-test-%j.out
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=10000
#SBATCH --partition=longrun
#SBATCH --gres=gpu:1

# EXPERIMENT NUMBER
exp_num=$1

# learning rate
lr=0.0001

# update step
us=50

# learning rate deacy
lrd=1.0

# dropout keep prob
kp=1.0

#batch size
bs=10

# optimizer
opt=adam

#########################
#########################

module load anaconda/3 cuda/8.0 cudnn/5.1

source activate tensorflow1.2_py2.7

python -u birnn_ctc_train.py $exp_num $lr $us $lrd $kp $bs $opt

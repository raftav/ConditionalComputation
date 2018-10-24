#!/bin/bash

#SBATCH --job-name=birnn
#SBATCH --error=tensorflow-test-%j.err
#SBATCH --output=tensorflow-test-%j.out
#SBATCH --ntasks=5
#SBATCH --ntasks-per-node=5
#SBATCH --mem=3000
#SBATCH --partition=longrun
#SBATCH --gres=gpu:1

# EXPERIMENT NUMBER
exp_num=$1

# learning rate
lr=0.001

# update step
us=3600

# learning rate deacy
lrd=1.0

# dropout keep prob
kp=1.0

#batch_size
bs=10

# optimizer
opt=adam

#########################
#########################

module load anaconda/3 cuda/8.0 cudnn/5.1

source activate tensorflow1.2_py2.7

python -u bi_gru_train.py $exp_num $lr $us $lrd $kp $bs $opt

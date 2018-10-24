#!/bin/bash

#SBATCH --job-name=Bi-hgru
#SBATCH --error=training-%j.err
#SBATCH --output=training-%j.out
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=20000
#SBATCH --partition=longrun
#SBATCH --gres=gpu:0

# EXPERIMENT NUMBER
exp_num=$1

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

# n hidden
nh=250

#########################
#########################

module load anaconda/3 cuda/8.0 cudnn/5.1

source activate tensorflow1.2_py2.7

python -u hgru_train_multireader.py $exp_num $lr $sar $us $lrd $kp $bs $opt $l2 $nh

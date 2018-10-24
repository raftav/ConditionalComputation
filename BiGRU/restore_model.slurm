#!/bin/bash

#SBATCH --job-name=brnn-re
#SBATCH --error=tensorflow-test-%j.err
#SBATCH --output=tensorflow-test-%j.out
#SBATCH --ntasks=5
#SBATCH --ntasks-per-node=5
#SBATCH --mem=3000
#SBATCH --partition=longrun
#SBATCH --gres=gpu:1

# EXPERIMENT NUMBER
exp_num=$1

#########################
#########################

module load anaconda/3 cuda/8.0 cudnn/5.1

source activate tensorflow1.2_py2.7

python -u bi_gru_restore_train.py $exp_num

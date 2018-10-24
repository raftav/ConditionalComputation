#!/bin/bash

#SBATCH --job-name=brnndev
#SBATCH --error=dev-%j.err
#SBATCH --output=dev-%j.out
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=3000
#SBATCH --partition=shortrun
#SBATCH --gres=gpu:1

# EXPERIMENT NUMBER
exp_num=1

#########################
#########################

module load anaconda/3 cuda/8.0 cudnn/5.1

source activate tensorflow1.2_py2.7

python -u bi_lstm_devset.py $exp_num

#!/bin/bash

#SBATCH --job-name=skipgru
#SBATCH --error=training-restore-%j.err
#SBATCH --output=training-restore-%j.out
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=5000
#SBATCH --partition=longrun
#SBATCH --gres=gpu:1

# EXPERIMENT NUMBER
exp_num=$1

#########################
#########################

module load anaconda/3 cuda/8.0 cudnn/5.1

source activate tensorflow1.2_py2.7

python -u skip_gru_restore_train_multireader.py $exp_num

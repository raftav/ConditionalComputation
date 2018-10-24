#!/bin/bash

#SBATCH --job-name=hgru-ev
#SBATCH --error=evaluation-%j.err
#SBATCH --output=evaluation-%j.out
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=3000
#SBATCH --partition=shortrun
#SBATCH --gres=gpu:1

# EXPERIMENT NUMBER
exp_num=1

res_epoch=5

#########################
#########################

module load anaconda/3 cuda/8.0 cudnn/5.1

source activate tensorflow1.2_py2.7

python -u hmlstm_evaluation.py $exp_num $res_epoch

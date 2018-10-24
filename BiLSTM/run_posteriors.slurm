#!/bin/bash

#SBATCH --job-name=birnn
#SBATCH --error=posteriors-%j.err
#SBATCH --output=posteriors-%j.out
#SBATCH --ntasks=5
#SBATCH --ntasks-per-node=5
#SBATCH --mem=3000
#SBATCH --partition=longrun
#SBATCH --gres=gpu:0


module load anaconda/3 cuda/8.0 cudnn/5.1

source activate tensorflow1.2_py2.7

python -u bi_lstm_posteriors.py

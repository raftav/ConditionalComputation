#!/bin/bash

#SBATCH --job-name=bigru-p
#SBATCH --error=posteriors-%j.err
#SBATCH --output=posteriors-%j.out
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --mem=3000
#SBATCH --partition=shortrun
#SBATCH --gres=gpu:0


module load anaconda/3 cuda/8.0 cudnn/5.1

source activate tensorflow1.2_py2.7

python -u hgru_posteriors.py

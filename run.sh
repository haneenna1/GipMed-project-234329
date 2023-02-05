#!/bin/sh
#SBATCH -p gipmed
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:2
#SBATCH -c 10
#SBATCH --output=sbatch_out/%j.out
#SBATCH --error=sbatch_out/%j.out

/home/haneenna/miniconda3/envs/GipMed/bin/python $@
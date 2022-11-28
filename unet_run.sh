#!/bin/sh
#SBATCH -p gipmed
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:4
#SBATCH -c 10
#SBATCH -o sbatch_out
#SBATCH -e sbatch_out

module load /home/haneenna/miniconda3/envs/GipMed/bin/python
/home/haneenna/miniconda3/envs/GipMed/bin/python /home/haneenna/GipMed-project-234329/main.py
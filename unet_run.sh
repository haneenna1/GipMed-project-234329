#!/bin/sh
#SBATCH -p gipmed
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:2
#SBATCH -c 10
#SBATCH -o sbatch.out
#SBATCH -e sbatch.out

/home/haneenna/miniconda3/envs/GipMed/bin/python /home/haneenna/GipMed-project-234329/main.py
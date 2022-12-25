#!/bin/sh
#SBATCH -p gipmed
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --job-name=400_mrkngs_trn
#SBATCH --output=sbatch_out/400_mrkngs_trn.out
#SBATCH --error=sbatch_out/400_mrkngs_trn.out
/home/haneenna/miniconda3/envs/GipMed/bin/python /home/haneenna/GipMed-project-234329/main.py
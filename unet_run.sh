#!/bin/sh
#SBATCH -p gipmed
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --job-name=unet_bg_trn
#SBATCH --output=sbatch_out/unet_bg_trn.out
#SBATCH --error=sbatch_out/unet_job.err
/home/haneenna/miniconda3/envs/GipMed/bin/python /home/haneenna/GipMed-project-234329/main.py
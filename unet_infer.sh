#!/bin/sh
#SBATCH -p gipmed
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --job-name=TCG_added_markingss
#SBATCH --output=sbatch_out/tcg_mrkngss.out
#SBATCH --error=sbatch_out/tcg_mrkngss.out
/home/haneenna/miniconda3/envs/GipMed/bin/python /home/haneenna/GipMed-project-234329/inference.py
#!/bin/sh
#SBATCH -p gipmed
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:3
#SBATCH -c 10
#SBATCH -o sbatch.out
#SBATCH -e sbatch.out

/home/amir.bishara/miniconda3/envs/GipMed/bin/python /home/amir.bishara/workspace/project/final_repo/GipMed-project-234329/main.py

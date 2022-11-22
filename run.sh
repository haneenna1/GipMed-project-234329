#!/bin/sh
#SBATCH -p gipmed
#SBATCH --gres=gpu:2
#SBATCH -c 4      # cores requested
#SBATCH -o ./sbatch_out/%j.out
module load /home/haneenna/miniconda3/envs/GipMed/bin/python
python /home/haneenna/GipMed-project-234329/main.py
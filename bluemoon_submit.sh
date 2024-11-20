#!/bin/bash

#SBATCH --partition=bluemoon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --job-name=sfnn
#SBATCH --output=./vacc_out_files/%x_%j.out
#SBATCH --time=8:00:00

set -x

echo $1
echo $2

python3 run_exp.py "$1"
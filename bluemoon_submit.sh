#!/bin/bash

#SBATCH --partition=bluemoon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --job-name=sfnn
#SBATCH --output=./vacc_out_files/%x_%j.out
#SBATCH --time=8:00:00

set -x

conda activate sfnn-evo

echo $1
echo $2

python3 run_exp.py "$1"
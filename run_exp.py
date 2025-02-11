"""
Run an experiment
"""
import os
import argparse
import shutil
import json
import pickle
from datetime import datetime
import torch

from experiment import Experiment

# Parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("exp_file", type=str, help="Path to the experiment file")
parser.add_argument("--run_id", type=int, default=0, help="Run ID")
args = parser.parse_args()

# Load experiment
if args.exp_file[-5:] == ".json":
    # Create experiment directory
    os.makedirs('experiments', exist_ok=True)
    date = datetime.now().strftime("%Y-%m-%d")
    time = datetime.now().strftime("%H:%M:%S")
    exp_dir = f'./experiments/exp_{date}_{time}_run{args.run_id}'
    os.makedirs(exp_dir, exist_ok=True)

    # Move experiment file to experiment directory
    shutil.copy(args.exp_file, os.path.join(exp_dir, 'exp_config.json'))

    # Initialize new experiment from JSON config
    with open(args.exp_file, "r") as f:
        exp_config = json.load(f)
        experiment = Experiment(exp_dir=exp_dir,
                                ea=exp_config["ea"],
                                evo_parameters=exp_config["evo_parameters"])
elif args.exp_file[-4:] == ".pkl":
    # Load existing experiment from pickle file
    with open(args.exp_file, "rb") as f:
        experiment = pickle.load(f)
else:
    raise ValueError(f"Invalid experiment file: {args.exp_file}")

# Run experiment
if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    experiment.run()

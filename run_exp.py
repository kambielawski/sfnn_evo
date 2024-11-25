"""
Run an experiment
"""
import os
import argparse
import shutil
import json
from datetime import datetime
from evo import HillClimber, PopulationEA, RandomSearch
import torch

# Parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("exp_file", type=str, help="Path to the experiment file")
args = parser.parse_args()

# Load experiment configuration
with open(args.exp_file, "r") as f:
    exp_config = json.load(f)

# Create experiment directory
os.makedirs('experiments', exist_ok=True)
ea = exp_config["ea"]
date = datetime.now().strftime("%Y-%m-%d")

exp_dir = f'./experiments/exp_{date}_{ea}'
os.makedirs(exp_dir, exist_ok=True)

# Move experiment file to experiment directory
shutil.copy(args.exp_file, os.path.join(exp_dir, 'exp_config.json'))

# Run experiment
if __name__ == "__main__":
    torch.set_default_dtype(torch.float16)
    if exp_config["ea"] == "HillClimber":
        ea = HillClimber(exp_dir, **exp_config["evo_parameters"])
        ea.evolve()
    elif exp_config["ea"] == "PopulationEA":
        ea = PopulationEA(exp_dir, **exp_config["evo_parameters"])
        ea.evolve()
    elif exp_config["ea"] == "RandomSearch":
        ea = RandomSearch(exp_dir, **exp_config["evo_parameters"])
        ea.evolve()
    else:   
        raise ValueError(f"Invalid EA: {exp_config['ea']}")

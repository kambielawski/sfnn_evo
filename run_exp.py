"""
Run an experiment
"""
import os
import argparse
import shutil
import json
from datetime import datetime
from evo import HillClimber

# Parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("exp_file", type=str, help="Path to the experiment file")
args = parser.parse_args()

# Create experiment directory
os.makedirs('experiments', exist_ok=True)
date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
exp_dir = f'./experiments/exp_{date}'
os.makedirs(exp_dir, exist_ok=True)

# Move experiment file to experiment directory
shutil.copy(args.exp_file, os.path.join(exp_dir, 'exp_config.json'))

# Load experiment configuration
with open(args.exp_file, "r") as f:
    exp_config = json.load(f)

# Run experiment
if __name__ == "__main__":
    if exp_config["ea"] == "HillClimber":
        ea = HillClimber(exp_dir, **exp_config["evo_parameters"])
        ea.evolve()
    else:   
        raise ValueError(f"Invalid EA: {exp_config['ea']}")

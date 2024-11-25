"""
Run an experiment
"""
import os
import argparse
import shutil
import json
import pickle
from datetime import datetime
from evo import HillClimber, PopulationEA, RandomSearch
import torch

# Parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("pickled_ea", type=str, help="Path to the pickled EA file")
args = parser.parse_args()

# Load experiment configuration
with open(args.pickled_ea, "rb") as f:
    ea = pickle.load(f)

# Run experiment
if __name__ == "__main__":
    torch.set_default_dtype(torch.float16)
    ea.reevaluate_population()
    ea.evolve(continue_exp=True)

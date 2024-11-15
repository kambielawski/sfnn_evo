"""
Experiment visualization functions
"""
import pickle

import matplotlib.pyplot as plt
import numpy as np

def plot_best_fitness(exp_dir : str):
    with open(f'{exp_dir}/ea.pkl', 'rb') as f:
        ea = pickle.load(f)

    best_fitness_values = [individual.fitness for individual in ea.best_fitness_individuals]
    plt.plot(best_fitness_values)
    plt.show()

def plot_average_fitness(exp_dir : str):
    with open(f'{exp_dir}/ea.pkl', 'rb') as f:
        ea = pickle.load(f)

    average_fitness_values = [np.mean(fitness_values) for fitness_values in ea.population_fitness]
    plt.plot(average_fitness_values)
    plt.show()
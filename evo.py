"""
Evolutionary algorithms
"""

from abc import ABC, abstractmethod
from copy import deepcopy
import pickle
import os
import time

import numpy as np

from sfnn import SFNN
from rl import evaluate_sfnn, evaluate_sfnn_1env

################################################################################
# ABSTRACT BASE CLASSES
################################################################################

class Individual(ABC):
    """
    Abstract base class for individuals in an evolutionary algorithm
    """
    @abstractmethod
    def mutate(self, mutation_rate : float):
        pass

    @abstractmethod
    def crossover(self, other : 'Individual') -> 'Individual':
        pass

    @abstractmethod
    def evaluate(self):
        pass

################################################################################

class EvolutionaryAlgorithm(ABC):
    """
    Abstract base class for evolutionary algorithms
    """
    @abstractmethod
    def evolve(self):
        pass

    @abstractmethod
    def evolve_one_generation(self):
        pass

################################################################################
# CONCRETE IMPLEMENTATIONS
################################################################################

class SFNNIndividual(Individual):
    """
    Individual specification for a Structurally Flexible Neural Network
    """
    def __init__(self, 
                 neuron_size : int,
                 gru_size : int,
                 n_neurons : int,
                 lr : float,
                 ticks : int):
        self.sfnn = SFNN(neuron_size=neuron_size, 
                         n_neurons=n_neurons, 
                         lr=lr, 
                         ticks=ticks)
        self.neuron_size = neuron_size # Number of neurons in the FC input layer
        self.gru_size = gru_size       # Number of neurons in the GRU
        self.fitness = None
        self.genome_size = 0
        self.genome = self.sfnn.get_parameters()

    def mutate(self, mutation_rate : float):
        """
        Mutate every element in the genome with a probability of mutation_rate
        """
        for param_type in self.genome:
            for x in np.nditer(self.genome[param_type], op_flags=['readwrite']):
                if np.random.random() < mutation_rate:
                    x[...] = np.random.uniform(-1, 1)

        self.sfnn.set_parameters(self.genome)

    def crossover(self, other : 'SFNNIndividual') -> 'SFNNIndividual':
        """
        No crossover is performed
        """
        return self, other
    
    def evaluate(self):
        self.fitness = evaluate_sfnn_1env(self.sfnn)
    
################################################################################

class HillClimber(EvolutionaryAlgorithm):
    """
    Hill climber evolutionary algorithm
    """
    def __init__(self,
                 exp_dir : str,
                 population_size : int,
                 mutation_rate : float,
                 n_generations : int,
                 neuron_size : int,
                 gru_size : int,
                 n_neurons : int,
                 lr : float,
                 ticks : int):
        self.exp_dir = exp_dir
        
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.n_generations = n_generations
        self.neuron_size = neuron_size
        self.gru_size = gru_size

        # SFNN parameters
        self.n_neurons = n_neurons
        self.lr = lr
        self.ticks = ticks

        # Determined by environment
        self.output_layer_size = 2
        self.input_layer_size = 4

        self.population = []

        # Tracking
        self.best_fitness_individuals = []
        self.population_fitness = []

    def evolve(self):
        """
        Evolve the population
        """
        # Initialize population
        self.init_population()

        total_time = 0

        # Evolve the population
        for gen in range(self.n_generations):
            print(f'Generation {gen} running.')
            start_time = time.time()

            self.evolve_one_generation(gen)
            
            # Pickle the run
            if gen % 100 == 0:
                self.pickle_ea(self.exp_dir)

            # Tracking
            self.best_fitness_individuals.append(self.population[0])
            self.population_fitness.append([individual.fitness for individual in self.population])
            
            end_time = time.time()
            total_time += end_time - start_time
            print(f'Generation {gen}: {end_time - start_time} seconds. Average time per generation: {total_time / (gen + 1)}')

    def evolve_one_generation(self, gen : int):
        """
        Evolve the population for one generation
        """

        # Reproduction
        children = []
        for i in range(self.population_size):
            child = SFNNIndividual(self.neuron_size,
                                   self.gru_size,
                                   self.n_neurons,
                                   self.lr,
                                   self.ticks)
            child.genome = deepcopy(self.population[i].genome)
            child.mutate(self.mutation_rate)
            child.evaluate()
            children.append(child)

        # Selection
        for i in range(self.population_size):
            if self.population[i].fitness < children[i].fitness:
                self.population[i] = children[i]

        self.population.sort(key=lambda x: x.fitness, reverse=True)

        print(f'Best fitness: {self.population[0].fitness}')
        
    def init_population(self):
        """
        Initialize the population
        """
        self.population = [SFNNIndividual(self.neuron_size, 
                                          self.gru_size,
                                          self.n_neurons,
                                          self.lr,
                                          self.ticks) for _ in range(self.population_size)]
        # Evaluate the initial population
        for individual in self.population:
            individual.evaluate()

    def pickle_ea(self, exp_dir : str):
        """
        Pickle the EA
        """
        with open(os.path.join(exp_dir, 'ea.pkl'), 'wb') as f:
            pickle.dump(self, f)

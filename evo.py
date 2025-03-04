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
import torch

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
    def evolve_one_generation(self):
        pass

    @abstractmethod
    def init_population(self):
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
                 n_neurons : int,
                 lr : float,
                 ticks : int,
                 n_structures : int = 3):
        self.sfnn = SFNN(neuron_size=neuron_size,
                         n_neurons=n_neurons,
                         lr=lr,
                         ticks=ticks)
        self.n_structures = n_structures # Number of structures to evaluate
        self.neuron_size = neuron_size # Number of neurons in the FC input layer
        self.fitness = None
        self.episode_rewards = None
        self.genome_size = 0
        self.genome = self.sfnn.get_parameters()
        self.evaluated = False
        self.adjacency_matrices = None

    def mutate(self, mutation_rate : float):
        """
        Mutate every element in the genome with a probability of mutation_rate
        """
        for param_type in self.genome:
           for x in np.nditer(self.genome[param_type], op_flags=['readwrite']):
               if np.random.random() < mutation_rate:
                   x[...] = np.random.uniform(-1, 1)
        # for param_type in self.genome:
        #     self.genome[param_type] += torch.rand_like(self.genome[param_type])*torch.bernoulli(torch.ones_like(self.genome[param_type])*mutation_rate)

        self.sfnn.set_parameters(self.genome)

    def crossover(self, other : 'SFNNIndividual') -> 'SFNNIndividual':
        """
        No crossover is performed
        """
        return self, other
    
    def evaluate(self):
        self.fitness, self.episode_rewards, self.adjacency_matrices = evaluate_sfnn_1env(self.sfnn, n_structures=self.n_structures)
        self.evaluated = True
    
################################################################################

class HillClimber(EvolutionaryAlgorithm):
    """
    Hill climber evolutionary algorithm
    """
    def __init__(self,
                 exp_dir : str,
                 population_size : int,
                 mutation_rate : float,
                 neuron_size : int,
                 gru_size : int,
                 n_neurons : int,
                 lr : float,
                 ticks : int):
        self.exp_dir = exp_dir
        
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.neuron_size = neuron_size
        self.gru_size = gru_size

        # SFNN parameters
        self.n_neurons = n_neurons
        self.lr = lr
        self.ticks = ticks

        self.population = []

        # Tracking
        self.best_fitness_individuals = []
        self.population_fitness = []

        # Initialize population
        self.init_population()
        

    def evolve_one_generation(self, gen : int):
        """
        Evolve the population for one generation
        """

        # Reproduction
        children = []
        for i in range(self.population_size):
            child = SFNNIndividual(self.neuron_size,
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

################################################################################

class PopulationEA(EvolutionaryAlgorithm):
    """
    Population based EA
    """
    def __init__(self, 
                 exp_dir : str, 
                 population_size : int, 
                 mutation_rate : float, 
                 neuron_size : int, 
                 gru_size : int, 
                 n_neurons : int, 
                 lr : float, 
                 ticks : int,
                 tournament_size : int = 2,
                 n_structures : int = 1):
        self.exp_dir = exp_dir
        
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.neuron_size = neuron_size
        self.gru_size = gru_size
        self.n_structures = n_structures # Number of structures to evaluate

        # SFNN parameters
        self.n_neurons = n_neurons
        self.lr = lr
        self.ticks = ticks

        self.population = []
        self.generation = 0
        # Tracking
        self.best_fitness_individuals = []
        self.population_fitness = []

        self.init_population()

    # def evolve(self, continue_exp : bool = False):
    #     """
    #     Evolve the population
    #     """
    #     # Initialize population
    #     if not continue_exp:
    #         self.init_population()

    #     total_time = 0

    #     # Evolve the population
    #     while self.generation < self.n_generations+1:
    #         print(f'Generation {self.generation} running.')
    #         start_time = time.time()

    #         self.evolve_one_generation(self.generation)
            
    #         # Pickle the run
    #         if self.generation % 250 == 0:
    #             self.pickle_ea(self.exp_dir, gen=self.generation)

    #         # Tracking
    #         self.best_fitness_individuals.append(self.population[0])
    #         self.population_fitness.append([individual.fitness for individual in self.population])
            
    #         end_time = time.time()
    #         total_time += end_time - start_time
    #         self.generation += 1
    #         print(f'Generation {self.generation}: {end_time - start_time} seconds. Average time per generation: {total_time / self.generation}')

    def evolve_one_generation(self, gen : int):
        """
        Evolve the population for one generation
        """

        # Reproduction
        children = []
        for _ in range(self.population_size):
            # Tournament select parent
            parent = self.tournament_select(k=self.tournament_size)
            child = SFNNIndividual(self.neuron_size,
                                   self.n_neurons,
                                   self.lr,
                                   self.ticks,
                                   self.n_structures)
            child.genome = deepcopy(parent.genome)
            child.mutate(self.mutation_rate)
            child.evaluate()
            children.append(child)

        # Selection
        self.population.extend(children)
        self.population = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:self.population_size]

        print(f'Best fitness: {self.population[0].fitness}')

    def tournament_select(self, k : int) -> SFNNIndividual:
        """
        Tournament select k individuals from the population
        """
        tournament = np.random.choice(self.population, size=k, replace=False)
        return max(tournament, key=lambda x: x.fitness)
        
    def init_population(self):
        """
        Initialize the population
        """
        self.population = [SFNNIndividual(self.neuron_size, 
                                          self.n_neurons,
                                          self.lr,
                                          self.ticks,
                                          self.n_structures) for _ in range(self.population_size)]
        # Evaluate the initial population
        for individual in self.population:
            individual.evaluate()

    def reevaluate_population(self, n_structures : int = 1):
        """
        Reevaluate the population
        """
        for individual in self.population:
            individual.n_structures = n_structures
            individual.evaluate()

    def pickle_ea(self, exp_dir : str, gen : int):
        """
        Pickle the EA
        """
        with open(f'{exp_dir}/ea_{gen}.pkl', 'wb') as f:
            pickle.dump(self, f)

################################################################################

class RandomSearch(EvolutionaryAlgorithm):
    """
    Random search evolutionary algorithm
    """
    def __init__(self, 
                 exp_dir : str, 
                 population_size : int, 
                 mutation_rate : float, 
                 neuron_size : int, 
                 gru_size : int, 
                 n_neurons : int, 
                 lr : float, 
                 ticks : int,
                 tournament_size : int = 2,
                 n_structures : int = 1):
        self.exp_dir = exp_dir
        
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.neuron_size = neuron_size
        self.gru_size = gru_size

        # SFNN parameters
        self.n_neurons = n_neurons
        self.lr = lr
        self.ticks = ticks
        self.n_structures = n_structures

        self.population = []

        # Tracking
        self.best_fitness_individuals = []
        self.population_fitness = []

        self.init_population()


    def evolve_one_generation(self, gen : int):
        """
        Evolve the population for one generation
        """

        # Reproduction
        children = []
        for _ in range(self.population_size):
            child = SFNNIndividual(self.neuron_size,
                                   self.n_neurons,
                                   self.lr,
                                   self.ticks,
                                   self.n_structures)
            child.evaluate()
            children.append(child)

        # Selection
        self.population.extend(children)
        self.population = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:self.population_size]

        print(f'Best fitness: {self.population[0].fitness}')

    def init_population(self):
        """
        Initialize the population
        """
        self.population = [SFNNIndividual(self.neuron_size, 
                                          self.n_neurons,
                                          self.lr,
                                          self.ticks,
                                          self.n_structures) for _ in range(self.population_size)]
        # Evaluate the initial population
        for individual in self.population:
            individual.evaluate()

    def pickle_ea(self, exp_dir : str):
        """
        Pickle the EA
        """
        with open(os.path.join(exp_dir, 'ea.pkl'), 'wb') as f:
            pickle.dump(self, f)
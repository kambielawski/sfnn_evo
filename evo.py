"""
Evolutionary algorithms
"""

from abc import ABC, abstractmethod
from typing import Any
from copy import deepcopy

import numpy as np

from sfnn import SFNN
from rl import evaluate_sfnn

################################################################################
# ABSTRACT BASE CLASSES
################################################################################

class Individual(ABC):
    """
    Abstract base class for individuals in an evolutionary algorithm
    """
    @abstractmethod
    def mutate(self):
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
                 input_layer_size : int,
                 hidden_layer_size : int,
                 output_layer_size : int,
                 lr : float,
                 ticks : int):
        self.sfnn = SFNN(neuron_size=neuron_size, 
                         input_layer_size=input_layer_size, 
                         hidden_layer_size=hidden_layer_size, 
                         output_layer_size=output_layer_size, 
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

        # TODO: rewrite to operate on torch representation of genome
        return None

        # mutation

        # for _, array in self.genome.items():
        #     for x in np.nditer(array, op_flags=['readwrite']): # n-dimensional iterator
        #         if np.random.random() < mutation_rate:
        #             x[...] = x + np.random.normal(0, 0.2)

    def crossover(self, other : 'SFNNIndividual') -> 'SFNNIndividual':
        """
        No crossover is performed
        """
        return self, other
    
    def evaluate(self):
        self.fitness = evaluate_sfnn(self.sfnn)
    
################################################################################

class HillClimber(EvolutionaryAlgorithm):
    """
    Hill climber evolutionary algorithm
    """
    def __init__(self,
                 population_size : int,
                 mutation_rate : float,
                 n_generations : int,
                 neuron_size : int,
                 gru_size : int):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.n_generations = n_generations
        self.neuron_size = neuron_size
        self.gru_size = gru_size

        # SFNN parameters
        self.input_layer_size = 4
        self.hidden_layer_size = 10
        self.output_layer_size = 2
        self.lr = 0.1
        self.ticks = 2

        self.population = []

    def evolve(self):
        """
        Evolve the population
        """
        # Initialize population
        self.init_population()

        # Evolve the population
        for gen in range(self.n_generations):
            print(f'Generation {gen} running.')
            self.evolve_one_generation(gen)

    def evolve_one_generation(self, gen : int):
        """
        Evolve the population for one generation
        """

        # Reproduction
        children = []
        for i in range(self.population_size):
            child = SFNNIndividual(self.neuron_size, 
                                   self.gru_size,
                                   self.input_layer_size,
                                   self.hidden_layer_size,
                                   self.output_layer_size,
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
                                          self.input_layer_size,
                                          self.hidden_layer_size,
                                          self.output_layer_size,
                                          self.lr,
                                          self.ticks) for _ in range(self.population_size)]
        # Evaluate the initial population
        for individual in self.population:
            individual.evaluate()

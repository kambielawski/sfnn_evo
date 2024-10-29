"""
Evolutionary algorithms
"""

from abc import ABC, abstractmethod
from typing import Any
from copy import deepcopy

import numpy as np

from rl import evaluate_sfnn

################################################################################
# ABSTRACT BASE CLASSES
################################################################################

class Individual(ABC):
    """
    Abstract base class for individuals in an evolutionary algorithm
    """
    @abstractmethod
    def init_genome(self) -> Any:
        pass

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
                 gru_size : int):
        self.neuron_size = neuron_size # Number of neurons in the FC input layer
        self.gru_size = gru_size       # Number of neurons in the GRU
        self.genome = None
        self.fitness = None
        self.genome_size = 0
        self.init_genome()

    def init_genome(self) -> Any:
        self.genome = {
            'input_neuron_weights' : np.random.randn(self.neuron_size, self.neuron_size),
            'input_neuron_biases' : np.random.randn(self.neuron_size, self.neuron_size),
            'reservoir_neuron_weights' : np.random.randn(self.neuron_size, self.neuron_size),
            'reservoir_neuron_biases' : np.random.randn(self.neuron_size, self.neuron_size),
            'output_neuron_weights' : np.random.randn(self.neuron_size, self.neuron_size),
            'output_neuron_biases' : np.random.randn(self.neuron_size, self.neuron_size),
            'input_gru_weights' : np.random.randn(self.gru_size),
            'input_gru_biases' : np.random.randn(self.gru_size),
            'reservoir_gru_weights' : np.random.randn(self.gru_size),
            'reservoir_gru_biases' : np.random.randn(self.gru_size),
            'output_gru_weights' : np.random.randn(self.gru_size),
            'output_gru_biases' : np.random.randn(self.gru_size)
        }
        self.genome_size = sum(np.prod(self.genome[key].shape) for key in self.genome.keys())

    def mutate(self, mutation_rate : float):
        """
        Mutate every element in the genome with a probability of mutation_rate
        """
        for _, array in self.genome.items():
            for x in np.nditer(array, op_flags=['readwrite']): # n-dimensional iterator
                if np.random.random() < mutation_rate:
                    x[...] = x + np.random.normal(0, 0.2)

    def crossover(self, other : 'SFNNIndividual') -> 'SFNNIndividual':
        """
        No crossover is performed
        """
        return self, other
    
    def evaluate(self):
        self.fitness = evaluate_sfnn(self.genome)
    
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
            child = deepcopy(self.population[i])
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
        self.population = [SFNNIndividual(self.neuron_size, self.gru_size) for _ in range(self.population_size)]
        # Evaluate the initial population
        for individual in self.population:
            individual.evaluate()

import os
import time
import pickle

from evo import (
    HillClimber,
    PopulationEA,
    RandomSearch,
    EvolutionaryAlgorithm
)

class EAFactory:
    @staticmethod
    def create(ea_type: str, exp_dir: str, **parameters) -> EvolutionaryAlgorithm:
        ea_types = {
            "HillClimber": HillClimber,
            "PopulationEA": PopulationEA,
            "RandomSearch": RandomSearch
        }
        
        if ea_type not in ea_types:
            raise ValueError(f"Invalid EA type: {ea_type}")
            
        return ea_types[ea_type](exp_dir, **parameters)

class Experiment:
    def __init__(self,
                 exp_dir : str,
                 ea : str,
                 evo_parameters : dict):
        self.exp_dir = exp_dir
        self.evo_parameters = evo_parameters
        self.ea: EvolutionaryAlgorithm = EAFactory.create(ea, self.exp_dir, **self.evo_parameters)
        self.generation = 0

        # Data to track
        self.best_fitness_individuals = []
        self.population_fitness = []

    def run(self):
        # Evolve the EA
        while True:
            generation_start_time = time.time()

            # Evolve
            self.ea.evolve_one_generation(gen=self.generation)

            self.generation += 1

            # Tracking
            self.best_fitness_individuals.append(self.ea.population[0])
            self.population_fitness.append([individual.fitness for individual in self.ea.population])

            generation_end_time = time.time()
            print(f"Generation {self.generation} completed in {generation_end_time - generation_start_time} seconds")
            
            # Pickle every 5 generations
            if self.generation % 5 == 0:
                self.pickle()

    def pickle(self):
        with open(os.path.join(self.exp_dir, "experiment.pkl"), "wb") as f:
            pickle.dump(self, f)

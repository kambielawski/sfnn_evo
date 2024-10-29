from evo import HillClimber

if __name__ == '__main__':
    ea = HillClimber(population_size=1, 
                     mutation_rate=0.01, 
                     n_generations=100, 
                     neuron_size=10, 
                     gru_size=10)
    ea.evolve()

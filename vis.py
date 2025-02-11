"""
Experiment visualization functions
"""
import pickle
import networkx as nx
from sfnn import SFNN
from matplotlib.patches import Patch
import torch

import matplotlib.pyplot as plt
import numpy as np

def plot_best_fitness(pkl_file_path : str, label="Best Fitness", gens_start=None, gens_end=None):
    with open(pkl_file_path, 'rb') as f:
        ea = pickle.load(f)

    best_fitness_values = [individual.fitness for individual in ea.best_fitness_individuals]
    if gens_start is None or gens_end is None:
        plt.plot(best_fitness_values, label=label)
    else:
        plt.plot(range(gens_start, gens_end), best_fitness_values[gens_start:gens_end], label=label)
    plt.title('Best Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')

def plot_average_fitness(pkl_file_path : str, label="Average Fitness", gens=None):
    with open(pkl_file_path, 'rb') as f:
        ea = pickle.load(f)

    average_fitness_values = [np.mean(fitness_values) for fitness_values in ea.population_fitness]
    if gens is None:
        plt.plot(average_fitness_values, label=label)
    else:
        plt.plot(average_fitness_values[:gens], label=label)
    plt.title('Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')

def plot_episode_rewards(exp_dir : str, label="Episode Rewards"):
    with open(f'{exp_dir}/ea.pkl', 'rb') as f:
        ea = pickle.load(f)

    episode_rewards = ea.best_fitness_individuals[-1].episode_rewards
    plt.plot(range(1, len(episode_rewards)+1), episode_rewards, label=label)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

def best_individuals(exp_dir: str):
    with open(f'{exp_dir}/ea.pkl', 'rb') as f:
        ea = pickle.load(f)

    best = ea.best_fitness_individuals[-1]
    print(len(ea.best_fitness_individuals))
    print(best.episode_rewards)

    visualize_sfnn(best.sfnn, best.adjacency_matrices[0])
    visualize_sfnn(best.sfnn, best.adjacency_matrices[1])
    visualize_sfnn(best.sfnn, best.adjacency_matrices[2])


def visualize_best_sfnn(exp_dir: str):
    with open(f'{exp_dir}/ea.pkl', 'rb') as f:
        ea = pickle.load(f)

    sfnn = ea.best_fitness_individuals[-1].sfnn
    visualize_sfnn(sfnn)

def visualize_sfnn(sfnn : SFNN, adjacency_matrix : np.ndarray = None):
    if adjacency_matrix is None:
        adjacency_matrix = sfnn.Adjacency_matrix.detach().numpy()
    
    # Create a directed graph
    G = nx.DiGraph()
    num_nodes = adjacency_matrix.shape[0]
    input_size = sfnn.input_layer_size
    output_size = sfnn.output_layer_size
    num_hidden = num_nodes - input_size - output_size
    G.add_nodes_from(range(num_nodes))

    print(input_size, output_size, num_hidden)
    
    # Custom position layout
    pos = {}
    
    # Position input nodes in left column
    for i in range(input_size):
        pos[i] = (0, (input_size-1)/2 - i)
        
    # Position hidden nodes in middle, spread out in a grid
    hidden_cols = int(np.sqrt(num_hidden)) + 1  # number of columns in hidden layer grid
    hidden_rows = int(np.ceil(num_hidden / hidden_cols))
    for i in range(num_hidden):
        row = i // hidden_cols
        col = i % hidden_cols
        x = 0.5 + col * 0.25  # spread columns horizontally
        y = (hidden_rows-1)/2 - row  # center vertically
        pos[i + input_size] = (x, y)
    
    # Position output nodes in right column
    for i in range(output_size):
        pos[num_nodes - output_size + i] = (1.5, (output_size-1)/2 - i)
    
    # Add edges
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adjacency_matrix[i, j] != 0:
                G.add_edge(i, j, weight=adjacency_matrix[i, j])

    # Node colors
    node_colors = []
    for i in range(num_nodes):
        if i < input_size:
            node_colors.append('lightgreen')
        elif i >= num_nodes - output_size:
            node_colors.append('lightcoral')
        else:
            node_colors.append('lightblue')

    plt.figure(figsize=(12, 8))
    
    # Draw edges with weights affecting width
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, edge_color='gray',
                          width=[abs(w) * 2 for w in weights],
                          arrowsize=20)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=500)
    
    # Add labels
    nx.draw_networkx_labels(G, pos)
    
    # Add legend
    legend_elements = [
        Patch(facecolor='lightgreen', label='Input Nodes'),
        Patch(facecolor='lightblue', label='Hidden Nodes'),
        Patch(facecolor='lightcoral', label='Output Nodes')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title('SFNN Network Structure')
    plt.axis('off')

def plot_fitness_appended(ea_file_1: str, ea_file_2: str):
    with open(ea_file_1, 'rb') as f:
        ea_1 = pickle.load(f)
    with open(ea_file_2, 'rb') as f:
        ea_2 = pickle.load(f)

    fitness_1 = [individual.fitness for individual in ea_1.best_fitness_individuals]
    fitness_2 = [individual.fitness for individual in ea_2.best_fitness_individuals]

    fitness_appended = fitness_1 + fitness_2
    plt.plot(fitness_appended, label=f"{ea_file_1} and {ea_file_2}")
    plt.title('Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')

def populations_box_plot(pkl_file_paths):
    torch.set_default_dtype(torch.float16)
    evo_objects = [pickle.load(open(pkl_file_path, 'rb')) for pkl_file_path in pkl_file_paths]

    for evo_object in evo_objects:
        evo_object.reevaluate_population(n_structures=10)

    populations = [evo_object.population for evo_object in evo_objects]

    fitnesses = [[individual.fitness for individual in population] for population in populations]
    print(fitnesses)
    plt.boxplot(fitnesses, labels=['Gen ' + ''.join(filter(str.isdigit, pkl_file_path.split('/')[-1])) for pkl_file_path in pkl_file_paths])
    plt.title('Population Fitness')
    plt.xlabel('Population')
    plt.ylabel('Fitness')
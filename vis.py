"""
Experiment visualization functions
"""
import pickle
import networkx as nx
from sfnn import SFNN
from matplotlib.patches import Patch

import matplotlib.pyplot as plt
import numpy as np

def plot_best_fitness(exp_dir : str, label="Best Fitness", gens=None):
    with open(f'{exp_dir}/ea.pkl', 'rb') as f:
        ea = pickle.load(f)

    best_fitness_values = [individual.fitness for individual in ea.best_fitness_individuals]
    if gens is None:
        plt.plot(best_fitness_values, label=label)
    else:
        plt.plot(best_fitness_values[:gens], label=label)
    plt.title('Best Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')

def plot_average_fitness(exp_dir : str, label="Average Fitness", gens=None):
    with open(f'{exp_dir}/ea.pkl', 'rb') as f:
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

def visualize_sfnn(exp_dir: str):

    with open(f'{exp_dir}/ea.pkl', 'rb') as f:
        ea = pickle.load(f)

    sfnn = ea.best_fitness_individuals[-1].sfnn
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
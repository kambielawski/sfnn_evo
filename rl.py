"""
Implementation of RL gym environments
"""

from typing import Any
import random

import torch
import gymnasium as gym
import numpy as np
from sfnn import SFNN

def run_rl(policy : SFNN,
           env_name : str,
           n_episodes : int = 8) -> float:
    """
    Run a RL environment with a SFNN genome
    """

    env = gym.make(env_name)

    # Set up the SFNN parameters for this environment
    input_layer_size = int(np.prod(env.observation_space.shape))
    output_layer_size = int(np.prod(env.action_space.shape))
    policy.init_connectivity(input_layer_size, output_layer_size)

    # Reset the environment to generate the first observation
    seed = random.randint(0, 2**32 - 1)
    observation, _ = env.reset(seed=seed)
    reward = 0

    episode_rewards = []
    episode_count = 0  # Track number of episodes

    while episode_count < n_episodes:
        episode_reward = 0

        for t in range(1000):
            # Insert SFNN policy here
            # action = env.action_space.sample() # Random policy
            action = policy(torch.tensor(observation), torch.tensor(reward))
            observation, reward, terminated, truncated, _ = env.step(action.item())
            
            episode_reward += reward  # Accumulate reward
            
            if terminated or truncated:
                episode_count += 1
                episode_rewards.append(episode_reward)
                episode_reward = 0
                observation, _ = env.reset()
                break

    env.close()

    # Episode weighting... 
    weights = np.array(range(1, n_episodes+1)) / sum(range(1, n_episodes+1))
    episodes_weighted_avg = np.average(episode_rewards, weights=weights)

    print(f"Final reward: {episodes_weighted_avg}")

    return episodes_weighted_avg


def evaluate_sfnn(genome : Any) -> float:
    """
    Evaluate a SFNN genome and returns a fitness score
    """

    env_1_reward = run_rl(genome, "CartPole-v1")
    env_2_reward = run_rl(genome, "Acrobot-v1")

    final_reward = env_1_reward + env_2_reward

    return final_reward

if __name__ == "__main__":
    genome = SFNN(neuron_size=10,
                  n_neurons=16,
                  lr=0.1,
                  ticks=2)
    print(evaluate_sfnn(genome))

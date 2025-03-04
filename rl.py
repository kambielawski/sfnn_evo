"""
Implementation of RL gym environments
"""
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
    output_layer_size = int(env.action_space.n)
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
            action = policy(torch.tensor(observation, dtype=torch.float32), 
                           torch.tensor(reward, dtype=torch.float32))
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

    return episode_rewards, episodes_weighted_avg


def evaluate_sfnn(sfnn : SFNN) -> float:
    """
    Evaluate a SFNN genome and returns a fitness score
    """

    env_1_reward, env_1_weighted_avg = run_rl(sfnn, "CartPole-v1")
    env_2_reward, env_2_weighted_avg = run_rl(sfnn, "Acrobot-v1")
    env_3_reward, env_3_weighted_avg = run_rl(sfnn, "MountainCar-v0")

    env_1_reward_scaled = env_1_reward / 500.0  # CartPole-v1
    env_2_reward_scaled = (env_2_reward + 500.0) / 500.0  # Acrobot-v1
    env_3_reward_scaled = (env_3_reward + 200.0) / 200.0  # MountainCar-v0

    print(f"Rewards: {env_1_reward_scaled}, {env_2_reward_scaled}, {env_3_reward_scaled} = {env_1_reward_scaled + env_2_reward_scaled + env_3_reward_scaled}")

    final_reward = env_1_reward_scaled + env_2_reward_scaled + env_3_reward_scaled

    return final_reward

def evaluate_sfnn_1env(sfnn : SFNN, n_structures : int = 1) -> float:
    """
    Evaluate a SFNN genome and returns a fitness score
    """
    env_name = f"CartPole-v1"
    env = gym.make(env_name)
    input_layer_size = int(np.prod(env.observation_space.shape))
    output_layer_size = int(env.action_space.n)

    rewards = []
    rewards_weighted_avgs = []
    connectivities = []

    for _ in range(n_structures):
        # sfnn = sfnn.float()

        # Reset the SFNN for the next structure
        sfnn.init_connectivity(input_layer_size, output_layer_size)
        connectivities.append(sfnn.Adjacency_matrix.clone().detach().numpy())

        env_reward, env_weighted_avg = run_rl(sfnn, env_name)
        rewards.append(env_reward)
        rewards_weighted_avgs.append(env_weighted_avg)

    final_reward = np.average(rewards_weighted_avgs)
    print(f"Rewards: {rewards_weighted_avgs}, avg: {final_reward}")

    return final_reward, rewards_weighted_avgs, connectivities

if __name__ == "__main__":
    genome = SFNN(neuron_size=10,
                  n_neurons=16,
                  lr=0.1,
                  ticks=2)
    print(evaluate_sfnn(genome))

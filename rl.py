"""
Implementation of RL gym environments
"""

from typing import Any
import random

import torch

from sfnn import SFNN

import numpy as np
import gymnasium as gym

def run_rl(policy, env_name : str, n_episodes : int = 8) -> float:
    """
    Run a RL environment with a SFNN genome
    """

    env = gym.make(env_name)

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
                print(f"Episode {episode_count} finished with total reward: {episode_reward}")
                
                episode_reward = 0
                observation, _ = env.reset()
                break

    env.close()

    print(episode_rewards)

    return episode_rewards


def evaluate_sfnn(genome : Any) -> float:
    """
    Evaluate a SFNN genome and returns a fitness score
    """

    final_reward = run_rl(genome, "CartPole-v1")

    return final_reward

if __name__ == "__main__":
    genome = None
    print(evaluate_sfnn(genome))

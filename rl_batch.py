"""
Implementation of RL gym environments
"""
import random

import torch
import gymnasium as gym
import numpy as np
from sfnn import SFNN
from torch.profiler import profile, record_function, ProfilerActivity

def run_rl(policy : SFNN,
           env_name : str,
           n_episodes : int = 8) -> float:
    """
    Run a RL environment with a SFNN genome
    """

    env = gym.make_vec(env_name, num_envs=n_episodes)
    #env = gym.make(env_name)

    # Set up the SFNN parameters for this environment
    #input_layer_size = int(np.prod(env.observation_space.shape))
    #output_layer_size = int(np.prod(env.action_space.shape))
    input_layer_size = int(env.observation_space.shape[-1])
    output_layer_size = int(env.action_space[0].n)
    
    #policy.init_connectivity(input_layer_size, output_layer_size)
    policy.init_connectivity(input_layer_size, output_layer_size, batch=n_episodes)
    policy.eval()

    # Reset the environment to generate the first observation
    seed = random.randint(0, 2**32 - 1)
    observation, _ = env.reset(seed=seed)
    reward = np.zeros(n_episodes)

    #episode_rewards = []
    #episode_count = 0  # Track number of episodes

    terminated_episodes = np.array([], dtype=int)
    episode_rewards = torch.zeros(n_episodes, dtype=torch.float16)
    
    for _ in range(1000):
        
        action = policy(torch.tensor(observation, dtype=torch.float16), torch.tensor(reward, dtype=torch.float16).view(n_episodes, 1))
        observation, reward, terminated, truncated, _ = env.step(action.numpy())
        
        terminated_episodes = np.unique(np.append(terminated_episodes, terminated.nonzero()))
        
        observation[terminated_episodes] = 0
        reward[terminated_episodes] = 0
        episode_rewards += reward

        if len(terminated_episodes) == n_episodes:
            break    

    #while episode_count < n_episodes:
    #    episode_reward = 0
    #    for t in range(1000):
    #        # Insert SFNN policy here
    #        with torch.no_grad():
    #            #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
    #                #with record_function("model_inference"):
    #            #print(torch.tensor(observation, dtype=torch.float16).shape, reward.shape)
    #            action = policy(torch.tensor(observation, dtype=torch.float16), torch.tensor(reward, dtype=torch.float16))
    #            
    #            #action = env.action_space.sample()
    #            
    #        #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=1000))
    #        #print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=1000))
    #        observation, reward, terminated, truncated, _ = env.step(action.item())
    #        
    #        episode_reward += reward  # Accumulate reward
    #        
    #        if terminated or truncated:
    #            episode_count += 1
    #            episode_rewards.append(episode_reward)
    #            episode_reward = 0
    #            observation, _ = env.reset()
    #            break
    
    env.close()

    # Episode weighting... 
    weights = np.array(range(1, n_episodes+1)) / sum(range(1, n_episodes+1))
    episodes_weighted_avg = np.average(episode_rewards, weights=weights)

    return episodes_weighted_avg


def evaluate_sfnn(sfnn : SFNN) -> float:
    """
    Evaluate a SFNN genome and returns a fitness score
    """

    env_1_reward = run_rl(sfnn, "CartPole-v1")
    env_2_reward = run_rl(sfnn, "Acrobot-v1")
    env_3_reward = run_rl(sfnn, "MountainCar-v0")

    env_1_reward_scaled = env_1_reward / 500.0  # CartPole-v1
    env_2_reward_scaled = (env_2_reward + 500.0) / 500.0  # Acrobot-v1
    env_3_reward_scaled = (env_3_reward + 200.0) / 200.0  # MountainCar-v0

    print(f"Rewards: {env_1_reward_scaled}, {env_2_reward_scaled}, {env_3_reward_scaled} = {env_1_reward_scaled + env_2_reward_scaled + env_3_reward_scaled}")

    final_reward = env_1_reward_scaled + env_2_reward_scaled + env_3_reward_scaled

    return final_reward

def evaluate_sfnn_1env(sfnn : SFNN) -> float:
    """
    Evaluate a SFNN genome and returns a fitness score
    """

    env_1_reward = run_rl(sfnn, "CartPole-v1")
    env_1_reward_scaled = env_1_reward / 500.0  # CartPole-v1
    print(f"Rewards: {env_1_reward_scaled}")

    return env_1_reward_scaled

if __name__ == "__main__":
    genome = SFNN(neuron_size=10,
                  n_neurons=16,
                  lr=0.1,
                  ticks=2)
    print(evaluate_sfnn(genome))

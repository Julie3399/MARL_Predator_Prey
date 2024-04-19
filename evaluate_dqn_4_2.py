from utils.load_model import load_dqn
from baseline.DQN import DQNAgent
from multiagent.mpe.predator_prey import predator_prey_2
from multiagent.mpe._mpe_utils.simple_env import SimpleEnv
import numpy as np
import wandb
import os
from collections import deque


save_dir = 'DQN_DQN_models_3_1'

env = predator_prey_2.parallel_env(render_mode="rgb_array", max_cycles=25)
observations, infos = env.reset()

dqn_agent_predator_0 = DQNAgent(state_size=env.observation_space("predator_0").shape[0], action_size=env.action_space("predator_0").n, seed=11)
dqn_agent_predator_1 = DQNAgent(state_size=env.observation_space("predator_1").shape[0], action_size=env.action_space("predator_1").n, seed=22)
dqn_agent_predator_2 = DQNAgent(state_size=env.observation_space("predator_2").shape[0], action_size=env.action_space("predator_2").n, seed=33)
# dqn_agent_predator_3 = DQNAgent(state_size=env.observation_space("predator_3").shape[0], action_size=env.action_space("predator_3").n, seed=44)

dqn_agent_prey_0 = DQNAgent(state_size=env.observation_space("prey_0").shape[0], action_size=env.action_space("prey_0").n, seed=45)
# dqn_agent_prey_1 = DQNAgent(state_size=env.observation_space("prey_1").shape[0], action_size=env.action_space("prey_1").n, seed=46)

# Load the models for each agent
load_dqn(dqn_agent_predator_0, 'dqn_agent_predator_0', save_dir)
load_dqn(dqn_agent_predator_1, 'dqn_agent_predator_1', save_dir)
load_dqn(dqn_agent_predator_2, 'dqn_agent_predator_2', save_dir)
# load_dqn(dqn_agent_predator_3, 'dqn_agent_predator_3', save_dir)

load_dqn(dqn_agent_prey_0, 'dqn_agent_prey_0', save_dir)
# load_dqn(dqn_agent_prey_1, 'dqn_agent_prey_1', save_dir)

# Initialize wandb
wandb.init(project='Evaluate_Predator_3_Prey_1', name='DQN_DQN')

# # Set a folder to save the gifs
# gif_dir = 'DQN_gifs'
# if not os.path.exists(gif_dir):
#     os.makedirs(gif_dir)

eps = 0.01
# Define a window size for averaging episode rewards
WINDOW_SIZE = 20
episode_rewards_window = deque(maxlen=WINDOW_SIZE)

def evaluate_model(num_episodes):
    total_rewards = []
    collision_num = 0

    for episode in range(num_episodes):
        episode_rewards = []
        frames = []
        observations, _ = env.reset()
        agent_rewards = {'predator': [], 'prey': []}

        while env.agents:
            actions = {}
            # actions = act(obs)
            for agent, obs in observations.items():
                if "predator_0" in agent:
                    actions[agent] = dqn_agent_predator_0.act(obs, eps)
                elif "predator_1" in agent:
                    actions[agent] = dqn_agent_predator_1.act(obs, eps)
                elif "predator_2" in agent:
                    actions[agent] = dqn_agent_predator_2.act(obs, eps)
                # elif "predator_3" in agent:
                #     actions[agent] = dqn_agent_predator_3.act(obs, eps)
                elif "prey_0" in agent:
                    actions[agent] = dqn_agent_prey_0.act(obs, eps)
                # else:
                #     actions[agent] = dqn_agent_prey_1.act(obs, eps)

            # Take the chosen actions and observe the next state and rewards
            frames.append(env.render())
            next_observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # return collision number
            collision_num += env.unwrapped.get_all_benchmark_data()  
            
            # Store experiences and update
            for agent, obs in observations.items():
                action = actions[agent]
                reward = rewards[agent]
                next_obs = next_observations[agent]
                done = terminations[agent]

                if "predator_0" in agent:
                    agent_rewards['predator'].append(reward)                    
                    dqn_agent_predator_0.step(obs, action, reward, next_obs, done)
                elif "predator_1" in agent:
                    agent_rewards['predator'].append(reward)               
                    dqn_agent_predator_1.step(obs, action, reward, next_obs, done)
                elif "predator_2" in agent:
                    agent_rewards['predator'].append(reward)               
                    dqn_agent_predator_2.step(obs, action, reward, next_obs, done)
                # elif "predator_3" in agent:
                #     agent_rewards['predator'].append(reward)
                #     dqn_agent_predator_3.step(obs, action, reward, next_obs, done)
                elif "prey_0" in agent:
                    agent_rewards['prey'].append(reward)               
                    dqn_agent_prey_0.step(obs, action, reward, next_obs, done)
                # else:
                #     agent_rewards['prey'].append(reward)
                #     dqn_agent_prey_1.step(obs, action, reward, next_obs, done)

            episode_rewards.append(sum(rewards.values()))
            observations = next_observations

        # if episode % 20 == 0:
        #     SimpleEnv.display_frames_as_gif(frames, episode, gif_dir)

        # Calculate the mean reward for predators and prey
        mean_predator_reward = sum(agent_rewards['predator']) / len(agent_rewards['predator']) if agent_rewards['predator'] else 0
        mean_prey_reward = sum(agent_rewards['prey']) / len(agent_rewards['prey']) if agent_rewards['prey'] else 0

        # Log the mean rewards for predators and prey
        wandb.log({"Mean Predator Reward": mean_predator_reward})
        wandb.log({"Mean Prey Reward": mean_prey_reward})
        
        mean_one_episode_reward = sum(episode_rewards) / len(episode_rewards)
        total_rewards.append(mean_one_episode_reward)

        # Append the episode's cumulative reward to the window
        episode_rewards_window.append(sum(episode_rewards))
        mean_episode_reward = sum(episode_rewards_window) / len(episode_rewards_window)


        wandb.log({
            "Mean Episode Reward": mean_one_episode_reward,
            "Mean Episode Reward (Last {} episodes)".format(WINDOW_SIZE): mean_episode_reward
        })
        
        print("Collision Number So Far:", collision_num)

    avg_reward = np.mean(total_rewards)
    print(f'Average Reward over {num_episodes} episodes: {avg_reward}')
    wandb.finish()

evaluate_model(num_episodes=200)
env.close()
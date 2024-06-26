from utils.load_model import load_ddpg, load_maddpg
import wandb
from baseline.DQN import DQNAgent
from algorithm.DDPG import DDPG
from algorithm.MADDPG import MADDPG
import os
from collections import deque
from multiagent.mpe.predator_prey import predator_prey
from multiagent.mpe._mpe_utils.simple_env import SimpleEnv
import numpy as np
import imageio


save_dir = 'MADDPG_DDPG_models_3_1'

env = predator_prey.parallel_env(render_mode="rgb_array", max_cycles=25)
observations, infos = env.reset()

maddpg_agent = MADDPG(obs_dim=env.observation_space("predator_0").shape[0], act_dim=env.action_space("predator_0").n, num_predators = 3, hidden_size=128, seed=1)
ddpg_agent0 = DDPG(obs_dim=env.observation_space("prey_0").shape[0], act_dim=env.action_space("prey_0").n, hidden_size=128, seed=2)

# Load the models for each agent
load_maddpg(maddpg_agent, save_dir)
load_ddpg(ddpg_agent0, 'ddpg_agent0', save_dir)

# Initialize wandb
wandb.init(project='MAPP_evaluate_test', name='MADDPG_3_1')

# Set a folder to save the gifs
# gif_dir = 'MADDPG_gifs_5_3'
# if not os.path.exists(gif_dir):
#     os.makedirs(gif_dir)

def evaluate_model(num_episodes):
    total_rewards = []

    for episode in range(num_episodes):
        episode_rewards = []
        frames = []
        observations, _ = env.reset()
        while env.agents:
            actions = {}
            for agent, obs in observations.items():
                if "predator" in agent:
                    actions[agent] = maddpg_agent.act([obs])[0]
                else:
                    actions[agent] = ddpg_agent0.act(obs)
            
            frames.append(env.render())
            next_observations, rewards, terminations, truncations, infos = env.step(actions)

            # Store experiences and update
            for agent, obs in observations.items():
                reward = rewards[agent]
                next_obs = next_observations[agent]
                done = terminations[agent]
                
                if "predator" in agent:
                    maddpg_agent.store_experience(obs, actions[agent], reward, next_obs, done)
                    maddpg_losses = maddpg_agent.update()
                else:
                    ddpg_agent0.store_experience(obs, actions[agent], reward, next_obs, done)
                    ddpg_loss = ddpg_agent0.update()

            episode_rewards.append(sum(rewards.values()))
            observations = next_observations

        # if episode % 20 == 0:
        #     SimpleEnv.display_frames_as_gif(frames, episode, gif_dir)

        mean_one_episode_reward = sum(episode_rewards)/len(episode_rewards)
        total_rewards.append(mean_one_episode_reward)

        wandb.log({
            "Mean Episode Reward": mean_one_episode_reward
        })

    avg_reward = np.mean(total_rewards)
    print(f'Average Reward over {num_episodes} episodes: {avg_reward}')
    wandb.finish()

evaluate_model(num_episodes=200)
env.close()
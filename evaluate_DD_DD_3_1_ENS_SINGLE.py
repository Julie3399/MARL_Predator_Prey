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
from tqdm import tqdm


save_dir = 'DDPG_DDPG_models-ENS-SINGLE-3-1' # 'DDPG_DDPG_models_4_2_rewards(Far)'

env = predator_prey.parallel_env(render_mode="rgb_array", max_cycles=25)
observations, infos = env.reset()

ddpg_agent_predator_0 = DDPG(obs_dim=env.observation_space("predator_0").shape[0], act_dim=env.action_space("predator_0").n,
                   hidden_size=128, seed=2)
ddpg_agent_predator_1 = DDPG(obs_dim=env.observation_space("predator_1").shape[0], act_dim=env.action_space("predator_1").n,
                   hidden_size=128, seed=3)
ddpg_agent_predator_2 = DDPG(obs_dim=env.observation_space("predator_2").shape[0], act_dim=env.action_space("predator_2").n,
                   hidden_size=128, seed=4)
# ddpg_agent_predator_3 = DDPG(obs_dim=env.observation_space("predator_3").shape[0], act_dim=env.action_space("predator_3").n,
#                    hidden_size=128, seed=5)

ddpg_agent_prey_0 = DDPG(obs_dim=env.observation_space("prey_0").shape[0], act_dim=env.action_space("prey_0").n, hidden_size=128, seed=40)
# ddpg_agent_prey_1 = DDPG(obs_dim=env.observation_space("prey_1").shape[0], act_dim=env.action_space("prey_1").n, hidden_size=128, seed=41)

# Load the models for each agent
load_ddpg(ddpg_agent_predator_0, 'ddpg_agent_predator_0', save_dir)
load_ddpg(ddpg_agent_predator_1, 'ddpg_agent_predator_1', save_dir)
load_ddpg(ddpg_agent_predator_2, 'ddpg_agent_predator_2', save_dir)
# load_ddpg(ddpg_agent_predator_3, 'ddpg_agent_predator_3', save_dir)
load_ddpg(ddpg_agent_prey_0, 'ddpg_agent0', save_dir)
# load_ddpg(ddpg_agent_prey_1, 'ddpg_agent_prey_1', save_dir)

# Initialize wandb
wandb.init(project='Evaluate_Final_3_1', name='DDPG_DDPG_3_1_ENS_SINGLE')

# Set a folder to save the gifs
gif_dir = 'MADDPG_gifs_DD_DD_3_1'
if not os.path.exists(gif_dir):
    os.makedirs(gif_dir)

# Define a window size for averaging episode rewards
WINDOW_SIZE = 20
episode_rewards_window = deque(maxlen=WINDOW_SIZE)

def evaluate_model(num_episodes):
    total_rewards = []
    collision_num = 0

    for episode in tqdm(range(num_episodes), desc="Evaluating Episodes"):
        episode_rewards = []
        frames = []
        observations, _ = env.reset()
        agent_rewards = {'predator': [], 'prey': []}

        ddpg_predator_agents = {
            0: ddpg_agent_predator_0,
            1: ddpg_agent_predator_1,
            2: ddpg_agent_predator_2,
            # 3: ddpg_agent_predator_3
        }
        
        ddpg_prey_agents = {
            0: ddpg_agent_prey_0,
            # 1: ddpg_agent_prey_1,
        }

        while env.agents:
            actions = {}
            for agent, obs in observations.items():
                if "prey" in agent:
                    agent_id = int(agent.split("_")[1])
                    actions[agent] = ddpg_prey_agents[agent_id].act(obs)
                else:
                    # actions[agent] = ddpg_agent.act(obs)
                    agent_id = int(agent.split("_")[1])
                    actions[agent] = ddpg_predator_agents[agent_id].act(obs)

            frames.append(env.render())
            next_observations, rewards, terminations, truncations, infos = env.step(actions)
            # return collision number
            collision_num += env.unwrapped.get_all_benchmark_data()  
            
            # Store experiences and update
            for agent, obs in observations.items():
                reward = rewards[agent]
                next_obs = next_observations[agent]
                done = terminations[agent]

                if "prey" in agent:
                    agent_rewards['prey'].append(reward)
                    agent_id = int(agent.split("_")[1])
                    ddpg_agent = ddpg_prey_agents[agent_id]
                    ddpg_agent.store_experience(obs, actions[agent], reward, next_obs, done)

                    if agent_id == 0:
                        ddpg_loss0_prey = ddpg_agent_prey_0.update()
                    # elif agent_id == 1:
                    #     ddpg_loss1_prey = ddpg_agent_prey_1.update()
                else:
                    agent_rewards['predator'].append(reward)
                    agent_id = int(agent.split("_")[1])
                    ddpg_agent = ddpg_predator_agents[agent_id]
                    ddpg_agent.store_experience(obs, actions[agent], reward, next_obs, done)

                    if agent_id == 0:
                        ddpg_loss0 = ddpg_agent_predator_0.update()
                    elif agent_id == 1:
                        ddpg_loss1 = ddpg_agent_predator_1.update()
                    elif agent_id == 2:
                        ddpg_loss2 = ddpg_agent_predator_2.update()
                    # elif agent_id == 3:
                    #     ddpg_loss3 = ddpg_agent_predator_3.update()


            episode_rewards.append(sum(rewards.values()))
            observations = next_observations
            

        if episode % 20 == 0:
            SimpleEnv.display_frames_as_gif(frames, episode, gif_dir)

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

        # Compute the mean reward over the last N episodes
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
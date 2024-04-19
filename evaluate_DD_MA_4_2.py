from utils.load_model import load_ddpg, load_maddpg, load_maddpg_prey
import wandb
from baseline.DQN import DQNAgent
from algorithm.DDPG import DDPG
from algorithm.MADDPG import MADDPG
import os
from collections import deque
from multiagent.mpe.predator_prey import predator_prey_2
from multiagent.mpe._mpe_utils.simple_env import SimpleEnv
import numpy as np
import imageio
from tqdm import tqdm

save_dir = 'DDPG_MADDPG_models_3_1'

env = predator_prey_2.parallel_env(render_mode="rgb_array", max_cycles=25)
observations, infos = env.reset()

ddpg_agent_predator_0 = DDPG(obs_dim=env.observation_space("predator_0").shape[0], act_dim=env.action_space("predator_0").n,
                   hidden_size=128, seed=2)
ddpg_agent_predator_1 = DDPG(obs_dim=env.observation_space("predator_1").shape[0], act_dim=env.action_space("predator_1").n,
                   hidden_size=128, seed=3)
ddpg_agent_predator_2 = DDPG(obs_dim=env.observation_space("predator_2").shape[0], act_dim=env.action_space("predator_2").n,
                   hidden_size=128, seed=4)
ddpg_agent_predator_3 = DDPG(obs_dim=env.observation_space("predator_3").shape[0], act_dim=env.action_space("predator_3").n,
                   hidden_size=128, seed=5)
maddpg_agent_prey = MADDPG(obs_dim=env.observation_space("prey_0").shape[0], act_dim=env.action_space("prey_0").n,
                      num_predators=2, hidden_size=128, seed=1)
# ddpg_agent0 = DDPG(obs_dim=env.observation_space("prey_0").shape[0], act_dim=env.action_space("prey_0").n,
#                    hidden_size=128, seed=2, id=0)
# ddpg_agent1 = DDPG(obs_dim=env.observation_space("prey_0").shape[0], act_dim=env.action_space("prey_0").n,
#                    hidden_size=128, seed=2, id=1)
# ddpg_agent2 = DDPG(obs_dim=env.observation_space("prey_0").shape[0], act_dim=env.action_space("prey_0").n,
#                    hidden_size=128, seed=2, id=2)

# Load the models for each agent
load_ddpg(ddpg_agent_predator_0, 'ddpg_agent0', save_dir)
load_ddpg(ddpg_agent_predator_1, 'ddpg_agent1', save_dir)
load_ddpg(ddpg_agent_predator_2, 'ddpg_agent2', save_dir)
# load_ddpg(ddpg_agent_predator_3, 'ddpg_agent3', save_dir)
load_maddpg_prey(maddpg_agent_prey, save_dir)
# load_ddpg(ddpg_agent0, 'ddpg_agent0', save_dir)
# load_ddpg(ddpg_agent1, 'ddpg_agent1', save_dir)
# load_ddpg(ddpg_agent2, 'ddpg_agent2', save_dir)

# Initialize wandb
wandb.init(project='Evaluate_Predator_3_Prey_1', name='DDPG_MADDPG_3_1')

# Set a folder to save the gifs
# gif_dir = 'MADDPG_gifs_DD_MA_4_2'
# if not os.path.exists(gif_dir):
#     os.makedirs(gif_dir)

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
        
        
        while env.agents:
            actions = {}
            for agent, obs in observations.items():
                if "predator" in agent:
                    agent_id = int(agent.split("_")[1])
                    actions[agent] = ddpg_predator_agents[agent_id].act(obs)
                else:
                    actions[agent] = maddpg_agent_prey.act([obs])[0]

            frames.append(env.render())
            next_observations, rewards, terminations, truncations, infos = env.step(actions)
            # return collision number
            collision_num += env.unwrapped.get_all_benchmark_data()  
            
            # Store experiences and update
            for agent, obs in observations.items():
                reward = rewards[agent]
                next_obs = next_observations[agent]
                done = terminations[agent]

                if "predator" in agent:
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


                else:
                    agent_rewards['prey'].append(reward)
                    maddpg_agent_prey.store_experience(obs, actions[agent], reward, next_obs, done)
                    maddpg_losses_prey = maddpg_agent_prey.update()

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
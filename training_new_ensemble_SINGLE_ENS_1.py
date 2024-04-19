from multiagent.mpe.predator_prey import predator_prey_2
from algorithm.MADDPG import MADDPG
from algorithm.DDPG import DDPG
import wandb
from collections import deque
from utils.save_model import save_maddpg, save_ddpg
import os
from tqdm import tqdm
import random

env = predator_prey_2.parallel_env(render_mode="rgb_array", max_cycles=25)
observations, infos = env.reset()

# Initialize MADDPG agent for predators and DDPG agent for the prey
ddpg_agent_predator_0 = DDPG(obs_dim=env.observation_space("predator_0").shape[0], act_dim=env.action_space("predator_0").n,
                   hidden_size=128, seed=10)
ddpg_agent_predator_1 = DDPG(obs_dim=env.observation_space("predator_1").shape[0], act_dim=env.action_space("predator_1").n,
                   hidden_size=128, seed=20)
ddpg_agent_predator_2 = DDPG(obs_dim=env.observation_space("predator_2").shape[0], act_dim=env.action_space("predator_2").n,
                   hidden_size=128, seed=30)


ddpg_agent_predator_4 = DDPG(obs_dim=env.observation_space("predator_0").shape[0], act_dim=env.action_space("predator_0").n,
                   hidden_size=128, seed=11)
ddpg_agent_predator_5 = DDPG(obs_dim=env.observation_space("predator_1").shape[0], act_dim=env.action_space("predator_1").n,
                   hidden_size=128, seed=22)
ddpg_agent_predator_6 = DDPG(obs_dim=env.observation_space("predator_2").shape[0], act_dim=env.action_space("predator_2").n,
                   hidden_size=128, seed=33)

ddpg_agent_prey_1 = DDPG(obs_dim=env.observation_space("prey_0").shape[0], act_dim=env.action_space("prey_0").n,
                  hidden_size=128, seed=40)

# ddpg_agent_prey_2 = DDPG(obs_dim=env.observation_space("prey_0").shape[0], act_dim=env.action_space("prey_0").n,
#                   hidden_size=128, seed=50)


# Initialize wandb
wandb.init(project='Predator_3_Prey_1', name='DDPG-DDPG-ENS-SINGLE-3-1')

# Define the episode length
NUM_EPISODES = 5000

# Define a window size for averaging episode rewards
WINDOW_SIZE = 200
episode_rewards_window = deque(maxlen=WINDOW_SIZE)

# Initialize the maximum mean reward
max_mean_reward = float('-inf')


# Define the path where you want to save the models
save_dir = 'DDPG_DDPG_models-ENS-SINGLE-3-1'  # Make sure this directory exists or create it
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for episode in tqdm(range(NUM_EPISODES), desc="Training Episodes"):
    observations, _ = env.reset()
    episode_rewards = []
    p_list = [random.random(), random.random(), random.random()]
    while env.agents:
        actions = {}
        for agent, obs in observations.items():
            if "predator_0" in agent:
                if p_list[0] <= 0.5:

                    # actions[agent] = maddpg_agent.act([obs])[0]
                    actions[agent] = ddpg_agent_predator_0.act(obs)
                else:
                    actions[agent] = ddpg_agent_predator_4.act(obs)
            elif "predator_1" in agent:
                if p_list[1] <= 0.5:

                    # actions[agent] = maddpg_agent.act([obs])[0]
                    actions[agent] = ddpg_agent_predator_1.act(obs)
                else:
                    actions[agent] = ddpg_agent_predator_5.act(obs)
            elif "predator_2" in agent:
                if p_list[2] <= 0.5:

                    # actions[agent] = maddpg_agent.act([obs])[0]
                    actions[agent] = ddpg_agent_predator_2.act(obs)
                else:
                    actions[agent] = ddpg_agent_predator_6.act(obs)
            elif "prey_0" in agent:
                # if random.random() <= 0.5:
                actions[agent] = ddpg_agent_prey_1.act(obs)
                    # actions[agent] = ddpg_agent.act(obs)
                # else:
                #     actions[agent] = ddpg_agent_prey_2.act(obs)
                    # actions[agent] = ddpg_agent1.act(obs)

        next_observations, rewards, terminations, truncations, infos = env.step(actions)

        # Store experiences and update
        for agent, obs in observations.items():
            reward = rewards[agent]
            next_obs = next_observations[agent]
            done = terminations[agent]

            if "predator_0" in agent:
                if p_list[0] <= 0.5:
                    ddpg_agent_predator_0.store_experience(obs, actions[agent], reward, next_obs, done)
                    ddpg_loss_0 = ddpg_agent_predator_0.update()
                else:
                    ddpg_agent_predator_4.store_experience(obs, actions[agent], reward, next_obs, done)
                    ddpg_loss_0 = ddpg_agent_predator_4.update()
            elif "predator_1" in agent:
                if p_list[1] <= 0.5:
                    ddpg_agent_predator_1.store_experience(obs, actions[agent], reward, next_obs, done)
                    ddpg_loss_1 = ddpg_agent_predator_1.update()
                else:
                    ddpg_agent_predator_5.store_experience(obs, actions[agent], reward, next_obs, done)
                    ddpg_loss_1 = ddpg_agent_predator_5.update()
            elif "predator_2" in agent:
                if p_list[2] <= 0.5:
                    ddpg_agent_predator_2.store_experience(obs, actions[agent], reward, next_obs, done)
                    ddpg_loss_2 = ddpg_agent_predator_2.update()
                else:
                    ddpg_agent_predator_6.store_experience(obs, actions[agent], reward, next_obs, done)
                    ddpg_loss_2 = ddpg_agent_predator_6.update()
            elif "prey_0" in agent:
                # if random.random() <= 0.5:
                ddpg_agent_prey_1.store_experience(obs, actions[agent], reward, next_obs, done)
                ddpg_losses = ddpg_agent_prey_1.update()
                    # ddpg_agent.store_experience(obs, actions[agent], reward, next_obs, done)
                    # ddpg_loss = ddpg_agent.update()
                # else:
                #     # ddpg_agent1.store_experience(obs, actions[agent], reward, next_obs, done)
                #     # ddpg_loss = ddpg_agent1.update()
                #     ddpg_agent_prey_2.store_experience(obs, actions[agent], reward, next_obs, done)
                #     ddpg_losses = ddpg_agent_prey_2.update()

        episode_rewards.append(sum(rewards.values()))
        observations = next_observations

    # Calculate the mean reward for each episode by averaging over all the steps
    mean_one_episode_reward = sum(episode_rewards) / len(episode_rewards)

    # Append the episode's cumulative reward to the window
    episode_rewards_window.append(sum(episode_rewards))

    # Compute the mean reward over the last N episodes
    mean_episode_reward = sum(episode_rewards_window) / len(episode_rewards_window)


    if mean_episode_reward > max_mean_reward:
        max_mean_reward = mean_episode_reward
        save_ddpg(ddpg_agent_predator_0, 'ddpg_agent_predator_0', save_dir)
        save_ddpg(ddpg_agent_predator_1, 'ddpg_agent_predator_1', save_dir)
        save_ddpg(ddpg_agent_predator_2, 'ddpg_agent_predator_2', save_dir)
        save_ddpg(ddpg_agent_predator_4, 'ddpg_agent_predator_4', save_dir)
        save_ddpg(ddpg_agent_predator_5, 'ddpg_agent_predator_5', save_dir)
        save_ddpg(ddpg_agent_predator_6, 'ddpg_agent_predator_6', save_dir)
        save_ddpg(ddpg_agent_prey_1, 'ddpg_agent0', save_dir)
        # save_ddpg(ddpg_agent_prey_2, 'ddpg_agent1', save_dir)

        print(f"New best model saved at episode {episode}")


    # Log rewards and policy losses to wandb
    wandb.log({
        "Episode Reward": sum(episode_rewards),
        "Mean Episode Reward": mean_one_episode_reward,
        "Mean Episode Reward (Last {} episodes)".format(WINDOW_SIZE): mean_episode_reward,
        # "MADDPG Policy Loss (Predator 0)": maddpg_losses[0] if maddpg_losses and maddpg_losses[0] is not None else None,
        # "MADDPG Policy Loss (Predator 1)": maddpg_losses[1] if maddpg_losses and maddpg_losses[1] is not None else None,
        # "MADDPG Policy Loss (Predator 2)": maddpg_losses[2] if maddpg_losses and maddpg_losses[2] is not None else None,
        "DDPG Policy Loss (Predator 0)": ddpg_loss_0,
        "DDPG Policy Loss (Predator 1)": ddpg_loss_1,
        "DDPG Policy Loss (Predator 2)": ddpg_loss_2,
        "DDPG Policy Loss (Prey 0)": ddpg_losses
    })

    # Save the models in the last episode
    # if episode == NUM_EPISODES - 1:
    #     save_maddpg(maddpg_agent, save_dir)
    #     save_maddpg(maddpg_agent1, save_dir)
    #     save_ddpg(ddpg_agent, 'ddpg_agent', save_dir)
        # save_ddpg(ddpg_agent1, 'ddpg_agent1', save_dir)

# Finish the wandb run
wandb.finish()
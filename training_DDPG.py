import os
from multiagent.mpe.predator_prey import predator_prey_RS_1
from algorithm.DDPG import DDPG
import wandb
from collections import deque
from utils.save_model import save_ddpg
import os
from tqdm import tqdm

env = predator_prey_RS_1.parallel_env(render_mode="rgb_array", max_cycles=25)
observations, infos = env.reset()

# Initialize DDPG agent for predators and the prey
ddpg_agent_predator_0 = DDPG(obs_dim=env.observation_space("predator_0").shape[0], act_dim=env.action_space("predator_0").n, hidden_size=128, seed=10)
ddpg_agent_predator_1 = DDPG(obs_dim=env.observation_space("predator_1").shape[0], act_dim=env.action_space("predator_1").n, hidden_size=128, seed=20)
ddpg_agent_predator_2 = DDPG(obs_dim=env.observation_space("predator_2").shape[0], act_dim=env.action_space("predator_2").n, hidden_size=128, seed=30)
ddpg_agent_prey_0 = DDPG(obs_dim=env.observation_space("prey_0").shape[0], act_dim=env.action_space("prey_0").n, hidden_size=128, seed=40)

# Initialize wandb
wandb.init(project='Predator_3_Prey_1', name='DDPG_DDPG_3_1_____TEST_____')

# Define the episode length
NUM_EPISODES = 5000

# Define a window size for averaging episode rewards
WINDOW_SIZE = 200
episode_rewards_window = deque(maxlen=WINDOW_SIZE)

# Define the path where you want to save the models
save_dir = 'DDPG_DDPG_models_3_1______TEST______'  # Make sure this directory exists or create it
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# Initialize the maximum mean reward
max_mean_reward = float('-inf')


for episode in tqdm(range(NUM_EPISODES), desc="Training Episodes"):
    observations, _ = env.reset()
    episode_rewards = []

    while env.agents:
        actions = {}
        for agent, obs in observations.items():
            if "predator_0" in agent:
                actions[agent] = ddpg_agent_predator_0.act(obs)
            elif "predator_1" in agent:
                actions[agent] = ddpg_agent_predator_1.act(obs)
            elif "predator_2" in agent:
                actions[agent] = ddpg_agent_predator_2.act(obs)
            else:
                actions[agent] = ddpg_agent_prey_0.act(obs)
        
        next_observations, rewards, terminations, truncations, infos = env.step(actions)


        # Store experiences and update
        for agent, obs in observations.items():
            reward = rewards[agent]
            next_obs = next_observations[agent]
            done = terminations[agent]
            
            if "predator_0" in agent:
                ddpg_agent_predator_0.store_experience(obs, actions[agent], reward, next_obs, done)
                ddpg_losses_0 = ddpg_agent_predator_0.update()
            elif "predator_1" in agent:
                ddpg_agent_predator_1.store_experience(obs, actions[agent], reward, next_obs, done)
                ddpg_losses_1 = ddpg_agent_predator_1.update()
            elif "predator_2" in agent:
                ddpg_agent_predator_2.store_experience(obs, actions[agent], reward, next_obs, done)
                ddpg_losses_2 = ddpg_agent_predator_2.update()
            else:
                ddpg_agent_prey_0.store_experience(obs, actions[agent], reward, next_obs, done)
                ddpg_loss = ddpg_agent_prey_0.update()

        episode_rewards.append(sum(rewards.values()))
        observations = next_observations

    # Calculate the mean reward for each episode by averaging over all the steps
    mean_one_episode_reward = sum(episode_rewards)/len(episode_rewards)
    # Append the episode's cumulative reward to the window
    episode_rewards_window.append(sum(episode_rewards))

    # Compute the mean reward over the last N episodes
    mean_episode_reward = sum(episode_rewards_window) / len(episode_rewards_window)

    if mean_episode_reward > max_mean_reward:
        max_mean_reward = mean_episode_reward
        save_ddpg(ddpg_agent_predator_0, 'ddpg_agent_predator_0', save_dir)
        save_ddpg(ddpg_agent_predator_1, 'ddpg_agent_predator_1', save_dir)
        save_ddpg(ddpg_agent_predator_2, 'ddpg_agent_predator_2', save_dir)
        save_ddpg(ddpg_agent_prey_0, 'ddpg_agent_prey_0', save_dir)
        print(f"New best model saved at episode {episode}")

    # Log rewards and policy losses to wandb
    wandb.log({
        "Episode Reward": sum(episode_rewards),
        "Mean Episode Reward": mean_one_episode_reward,
        "Mean Episode Reward (Last {} episodes)".format(WINDOW_SIZE): mean_episode_reward,
        "DDPG Policy Loss (Predator 0)": ddpg_losses_0,
        "DDPG Policy Loss (Predator 1)": ddpg_losses_1,
        "DDPG Policy Loss (Predator 2)": ddpg_losses_2,
        "DDPG Policy Loss (Prey 0)": ddpg_loss
    })

    # Save the models in the last episode
    if episode == NUM_EPISODES - 1:
        save_ddpg(ddpg_agent_predator_0, 'ddpg_agent_predator_0', save_dir)
        save_ddpg(ddpg_agent_predator_1, 'ddpg_agent_predator_1', save_dir)
        save_ddpg(ddpg_agent_predator_2, 'ddpg_agent_predator_2', save_dir)
        save_ddpg(ddpg_agent_prey_0, 'ddpg_agent_prey_0', save_dir)

# Finish the wandb run
wandb.finish()
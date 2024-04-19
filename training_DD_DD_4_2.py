import os
from multiagent.mpe.predator_prey import predator_prey_RS
from algorithm.DDPG import DDPG
import wandb
from collections import deque
from utils.save_model import save_ddpg
import os
from tqdm import tqdm

env = predator_prey_RS.parallel_env(render_mode="rgb_array", max_cycles=25)
observations, infos = env.reset()

# Initialize DDPG agent for predators and the prey
ddpg_agent_predator_0 = DDPG(obs_dim=env.observation_space("predator_0").shape[0],
                             act_dim=env.action_space("predator_0").n, hidden_size=128, seed=10)
ddpg_agent_predator_1 = DDPG(obs_dim=env.observation_space("predator_1").shape[0],
                             act_dim=env.action_space("predator_1").n, hidden_size=128, seed=20)
ddpg_agent_predator_2 = DDPG(obs_dim=env.observation_space("predator_2").shape[0],
                             act_dim=env.action_space("predator_2").n, hidden_size=128, seed=30)
ddpg_agent_predator_3 = DDPG(obs_dim=env.observation_space("predator_3").shape[0],
                             act_dim=env.action_space("predator_3").n, hidden_size=128, seed=40)
ddpg_agent_prey_0 = DDPG(obs_dim=env.observation_space("prey_0").shape[0], act_dim=env.action_space("prey_0").n,
                         hidden_size=128, seed=50)
ddpg_agent_prey_1 = DDPG(obs_dim=env.observation_space("prey_1").shape[0], act_dim=env.action_space("prey_1").n,
                         hidden_size=128, seed=60)


# Initialize wandb
wandb.init(project='Predator_4_Prey_2', name='DDPG-DDPG_4_2_RS')

# Define the episode length
NUM_EPISODES = 5000

# Define a window size for averaging episode rewards
WINDOW_SIZE = 200
episode_rewards_window = deque(maxlen=WINDOW_SIZE)

# Define the path where you want to save the models
save_dir = 'DDPG_DDPG_models_4_2_RS'  # Make sure this directory exists or create it
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for episode in tqdm(range(NUM_EPISODES), desc="Training Episodes"):
    observations, _ = env.reset()
    episode_rewards = []
    agent_rewards = {agent: 0 for agent in env.agents}

    while env.agents:
        actions = {}
        for agent, obs in observations.items():
            if "predator_0" in agent:
                actions[agent] = ddpg_agent_predator_0.act(obs)
            elif "predator_1" in agent:
                actions[agent] = ddpg_agent_predator_1.act(obs)
            elif "predator_2" in agent:
                actions[agent] = ddpg_agent_predator_2.act(obs)
            elif "predator_3" in agent:
                actions[agent] = ddpg_agent_predator_3.act(obs)
            elif "prey_0" in agent:
                actions[agent] = ddpg_agent_prey_0.act(obs)
            elif "prey_1" in agent:
                actions[agent] = ddpg_agent_prey_1.act(obs)

        next_observations, rewards, terminations, truncations, infos = env.step(actions)

        # Store experiences and update
        for agent, obs in observations.items():
            reward = rewards[agent]
            wandb.log({f"{agent} Step Reward": reward})
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
            elif "predator_3" in agent:
                ddpg_agent_predator_3.store_experience(obs, actions[agent], reward, next_obs, done)
                ddpg_losses_3 = ddpg_agent_predator_3.update()
            elif "prey_0" in agent:
                ddpg_agent_prey_0.store_experience(obs, actions[agent], reward, next_obs, done)
                ddpg_loss = ddpg_agent_prey_0.update()
            elif "prey_1" in agent:
                ddpg_agent_prey_1.store_experience(obs, actions[agent], reward, next_obs, done)
                ddpg_loss_1 = ddpg_agent_prey_1.update()

        episode_rewards.append(sum(rewards.values()))
        observations = next_observations

    # Calculate the mean reward for each episode by averaging over all the steps
    mean_one_episode_reward = sum(episode_rewards) / len(episode_rewards)
    # Append the episode's cumulative reward to the window
    episode_rewards_window.append(sum(episode_rewards))

    # Compute the mean reward over the last N episodes
    mean_episode_reward = sum(episode_rewards_window) / len(episode_rewards_window)

    # Log rewards and policy losses to wandb
    wandb.log({
        "Episode Reward": sum(episode_rewards),
        "Mean Episode Reward": mean_one_episode_reward,
        "Mean Episode Reward (Last {} episodes)".format(WINDOW_SIZE): mean_episode_reward,
        "DDPG Policy Loss (Predator 0)": ddpg_losses_0,
        "DDPG Policy Loss (Predator 1)": ddpg_losses_1,
        "DDPG Policy Loss (Predator 2)": ddpg_losses_2,
        "DDPG Policy Loss (Predator 3)": ddpg_losses_3,
        "DDPG Policy Loss (Prey 0)": ddpg_loss,
        "DDPG Policy Loss (Prey 1)": ddpg_loss_1
    })

    # Save the models in the last episode
    if episode == NUM_EPISODES - 1:
        save_ddpg(ddpg_agent_predator_0, 'ddpg_agent_predator_0', save_dir)
        save_ddpg(ddpg_agent_predator_1, 'ddpg_agent_predator_1', save_dir)
        save_ddpg(ddpg_agent_predator_2, 'ddpg_agent_predator_2', save_dir)
        save_ddpg(ddpg_agent_predator_3, 'ddpg_agent_predator_3', save_dir)
        save_ddpg(ddpg_agent_prey_0, 'ddpg_agent_prey_0', save_dir)
        save_ddpg(ddpg_agent_prey_1, 'ddpg_agent_prey_1', save_dir)

# Finish the wandb run
wandb.finish()
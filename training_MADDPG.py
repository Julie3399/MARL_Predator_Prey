from multiagent.mpe.predator_prey import predator_prey
from algorithm.MADDPG import MADDPG
from algorithm.DDPG import DDPG
import wandb
from collections import deque
from utils.save_model import save_maddpg, save_ddpg
import os
from tqdm import tqdm

env = predator_prey.parallel_env(render_mode="rgb_array", max_cycles=25)
observations, infos = env.reset()

# Initialize MADDPG agent for predators and DDPG agent for the prey
maddpg_agent = MADDPG(obs_dim=env.observation_space("predator_0").shape[0], act_dim=env.action_space("predator_0").n, num_predators = 3, hidden_size=128, seed=10)
ddpg_agent0 = DDPG(obs_dim=env.observation_space("prey_0").shape[0], act_dim=env.action_space("prey_0").n, hidden_size=128, seed=20)
# ddpg_agent1 = DDPG(obs_dim=env.observation_space("prey_1").shape[0], act_dim=env.action_space("prey_1").n, hidden_size=128, seed=21)
# ddpg_agent2 = DDPG(obs_dim=env.observation_space("prey_2").shape[0], act_dim=env.action_space("prey_2").n, hidden_size=128, seed=22)

# Initialize wandb
wandb.init(project='Predator_3_Prey_1', name='MADDPG-DDPG_3_1')

# Define the episode length
NUM_EPISODES = 5000

# Define a window size for averaging episode rewards
WINDOW_SIZE = 200
episode_rewards_window = deque(maxlen=WINDOW_SIZE)

# Define the path where you want to save the models
save_dir = 'MADDPG_DDPG_models_3_1'  # Make sure this directory exists or create it
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Initialize the maximum mean reward
max_mean_reward = float('-inf')

for episode in tqdm(range(NUM_EPISODES), desc="Training Episodes"):
    observations, _ = env.reset()
    episode_rewards = []
    # agent_rewards = {agent: 0 for agent in env.agents}

    while env.agents:
        actions = {}
        for agent, obs in observations.items():
            if "predator" in agent:
                actions[agent] = maddpg_agent.act([obs])[0]
            else:
                actions[agent] = ddpg_agent0.act(obs)
        
        next_observations, rewards, terminations, truncations, infos = env.step(actions)

        # Store experiences and update
        for agent, obs in observations.items():
            reward = rewards[agent]
            # wandb.log({f"{agent} Step Reward": reward})
            # agent_rewards[agent] += reward
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

    # Calculate the mean reward for each episode by averaging over all the steps
    mean_one_episode_reward = sum(episode_rewards)/len(episode_rewards)

    # Append the episode's cumulative reward to the window
    episode_rewards_window.append(sum(episode_rewards))

    # Compute the mean reward over the last N episodes
    mean_episode_reward = sum(episode_rewards_window) / len(episode_rewards_window)

    # Save every 1000 episodes
    if (episode + 1) % 100 == 0:
        save_maddpg(maddpg_agent, save_dir)
        save_ddpg(ddpg_agent0, 'ddpg_agent0', save_dir)
        # save_ddpg(ddpg_agent1, 'ddpg_agent1', save_dir)
        # save_ddpg(ddpg_agent2, 'ddpg_agent2', save_dir)


    # Save the best model if the new mean reward is higher than the previous max
    if mean_episode_reward > max_mean_reward:
        max_mean_reward = mean_episode_reward
        save_maddpg(maddpg_agent, save_dir)
        save_ddpg(ddpg_agent0, 'ddpg_agent0', save_dir)
        # save_ddpg(ddpg_agent1, 'ddpg_agent1', save_dir)
        # save_ddpg(ddpg_agent2, 'ddpg_agent2', save_dir)
        print(f"New best model saved at episode {episode}")

    # Log rewards and policy losses to wandb
    for i in range(len(maddpg_losses)):
        wandb.log({
            f"MADDPG Policy Loss (Predator {i})": maddpg_losses[i] if maddpg_losses and maddpg_losses[
                i] is not None else None,
        })
    # wandb.log({"Agent {}: Episode Reward".format(agent): agent_rewards[agent] for agent in env.agents})
    wandb.log({
        "Episode Reward": sum(episode_rewards),
        "Mean Episode Reward": mean_one_episode_reward,
        "Mean Episode Reward (Last {} episodes)".format(WINDOW_SIZE): mean_episode_reward,
        # "MADDPG Policy Loss (Predator 0)": maddpg_losses[0] if maddpg_losses and maddpg_losses[0] is not None else None,
        # "MADDPG Policy Loss (Predator 1)": maddpg_losses[1] if maddpg_losses and maddpg_losses[1] is not None else None,
        # "MADDPG Policy Loss (Predator 2)": maddpg_losses[2] if maddpg_losses and maddpg_losses[2] is not None else None,
        "DDPG Policy Loss (Prey 0)": ddpg_loss,
        # "DDPG Policy Loss (Prey 1)": ddpg_loss1,
        # "DDPG Policy Loss (Prey 2)": ddpg_loss2
    })

    # Save the models in the last episode
    if episode == NUM_EPISODES - 1:
        save_maddpg(maddpg_agent, save_dir)
        save_ddpg(ddpg_agent0, 'ddpg_agent0', save_dir)
        # save_ddpg(ddpg_agent1, 'ddpg_agent1', save_dir)
        # save_ddpg(ddpg_agent2, 'ddpg_agent2', save_dir)

# Finish the wandb run
wandb.finish()
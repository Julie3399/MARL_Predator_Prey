import torch
import torch.nn.functional as F
from utils.network import PolicyNetwork, MADDPGQNetwork, DDPGQNetwork
from utils.ReplayBuffer import ReplayBuffer
import random
import numpy as np

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG_prey:
    def __init__(self, obs_dim, act_dim, num_predators, hidden_size, seed, buffer_size=100000, batch_size=64, LR=0.001):
        self.num_predators = num_predators
        self.act_dim = act_dim
        self.seed = seed
        self.hidden_size = hidden_size
        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)
        self.LR = LR
        # Define policy(actor) and Q-networks(critic) for each predator
        self.predator_policy_nets = [PolicyNetwork(obs_dim, act_dim, seed, hidden_size).to(device) for _ in range(num_predators)]
        # self.predator_q_net = MADDPGQNetwork(obs_dim, act_dim, num_predators, seed, hidden_size).to(device)
        # self.predator_q_nets = [DDPGQNetwork(obs_dim, act_dim, seed, hidden_size).to(device) for _ in range(num_predators)]
        self.predator_q_nets = [MADDPGQNetwork(obs_dim, act_dim, num_predators, seed, hidden_size).to(device) for _ in range(num_predators)]

        # Define optimizers for each predator's policy network and the shared Q-network
        self.predator_policy_optimizers = [torch.optim.Adam(net.parameters(), lr=LR, weight_decay=0.0001) for net in self.predator_policy_nets]
        # self.predator_q_optimizer = torch.optim.Adam(self.predator_q_net.parameters(), lr=LR, weight_decay=0.0001)
        self.predator_q_optimizers = [torch.optim.Adam(net.parameters(), lr=LR, weight_decay=0.0001) for net in self.predator_q_nets]

        # Define target networks for each predator and the shared Q-network
        self.target_predator_policy_nets = [PolicyNetwork(obs_dim, act_dim, seed, hidden_size) for _ in range(num_predators)]
        # self.target_predator_q_nets = [DDPGQNetwork(obs_dim, act_dim, seed, hidden_size) for _ in range(num_predators)]
        self.target_predator_q_nets = [MADDPGQNetwork(obs_dim, act_dim, num_predators, seed, hidden_size) for _ in range(num_predators)]

        # Initialize target network weights to match the original networks
        for i in range(num_predators):
            self.target_predator_policy_nets[i].load_state_dict(self.predator_policy_nets[i].state_dict())
            self.target_predator_q_nets[i].load_state_dict(self.predator_q_nets[i].state_dict())
        
    def act(self, observations, epsilon=0.05):
        """Choose actions for all predators."""
        actions = []
        with torch.no_grad():
            for i, obs in enumerate(observations):
                if random.random() < epsilon:
                    # Exploration: choose a random action
                    action = random.choice(np.arange(self.act_dim))
                else:
                    # Exploitation: choose the best action according to the policy
                    action_probs = self.predator_policy_nets[i](torch.tensor(obs, dtype=torch.float32))
                    action = action_probs.argmax().item()
                actions.append(action)
        return actions
    
    def store_experience(self, state, action, reward, next_state, done):
        """Stores experience in the replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update(self):
        """Perform MADDPG learning update for predators and return policy losses."""
        # policy_losses = [None, None, None]
        policy_losses = [None] * self.num_predators
        if len(self.replay_buffer) < self.replay_buffer.batch_size:
            return policy_losses

        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)

        # Convert actions to one-hot encoded format and get next actions
        actions_one_hot = torch.zeros(actions.size(0), self.act_dim)
        actions_one_hot.scatter_(1, actions.unsqueeze(-1).long(), 1)
        # print(self.predator_policy_nets[0],next_states.shape)
        next_actions = torch.stack([net(next_states) for net in self.predator_policy_nets], dim=1).view(actions.size(0), -1)

        # Compute target Q-values
        for i, (target_q_net, q_net, optimizer) in enumerate(zip(self.target_predator_q_nets, self.target_predator_q_nets,self.predator_q_optimizers)):
            target_q_values = target_q_net(next_states.repeat(1, self.num_predators), next_actions)
            y = rewards[i] + (1 - dones[i]) * 0.99 * target_q_values.squeeze() 

            # Compute Q-network loss
            q_values = q_net(states.repeat(1, self.num_predators), actions_one_hot.repeat(1, self.num_predators)).squeeze() #?
            q_loss = F.mse_loss(q_values, y.detach())

            # Update Q-network
            optimizer.zero_grad()
            q_loss.backward()
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0) 
            optimizer.step()

        # Update each predator's policy network
        for i, (policy_net, optimizer) in enumerate(zip(self.predator_policy_nets, self.predator_policy_optimizers)):
            # Compute policy loss
            curr_actions = policy_net(states)
            other_actions = torch.cat([self.predator_policy_nets[j](states) if j != i else curr_actions for j in range(self.num_predators)], dim=1)#.reshape(states.shape[0],self.num_predators,-1)
            loss = -self.predator_q_nets[i](states.repeat(1, self.num_predators), other_actions).mean()

            # Store the policy loss for this predator
            policy_losses[i] = loss.detach().item()

            # Update policy network
            optimizer.zero_grad()
            loss.backward()
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
            optimizer.step()

        # Soft update of the target networks
        for policy_net, target_policy_net in zip(self.predator_policy_nets, self.target_predator_policy_nets):
            self.soft_update(policy_net, target_policy_net, tau=0.005)

        for q_net, target_q_net in zip(self.predator_q_nets,self.target_predator_q_nets):
            self.soft_update(q_net, target_q_net, tau=0.005)

        return policy_losses

    @staticmethod
    def soft_update(local_model, target_model, tau=0.01):
        """Soft update model parameters: θ_target = τ*θ_local + (1 - τ)*θ_target"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

  
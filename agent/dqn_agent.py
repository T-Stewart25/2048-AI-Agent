# agent/dqn_agent.py

import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Define possible actions
ACTIONS = ['up', 'down', 'left', 'right']

# Determine the device to run the model on (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def state_to_tensor(board):
    """
    Converts a 4x4 NumPy board into a PyTorch Tensor.
    Applies a logarithmic transformation to handle zero values.
    """
    transformed = np.log2(board + 1)  # Avoid log(0) by adding 1
    return torch.tensor(transformed, dtype=torch.float32).view(1, -1).to(device)

class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network Architecture.
    Separates the estimation of the state-value and advantage for each action.
    """
    def __init__(self, input_size=16, hidden_size=128, output_size=4):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()

        # Value stream
        self.fc_value = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, 1)

        # Advantage stream
        self.fc_advantage = nn.Linear(hidden_size, hidden_size)
        self.advantage = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass through the network.
        Combines value and advantage streams to compute Q-values.
        """
        x = self.relu(self.fc1(x))

        # Value stream
        val = self.relu(self.fc_value(x))
        val = self.value(val)

        # Advantage stream
        adv = self.relu(self.fc_advantage(x))
        adv = self.advantage(adv)

        # Combine value and advantage to get Q-values
        q_vals = val + (adv - adv.mean(dim=1, keepdim=True))
        return q_vals

class ReplayBuffer:
    """
    Experience Replay Buffer to store and sample experiences.
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Stores an experience tuple in the buffer.
        Converts states to Tensors if they are NumPy arrays.
        """
        if isinstance(state, np.ndarray):
            state = state_to_tensor(state)
        if isinstance(next_state, np.ndarray):
            next_state = state_to_tensor(next_state)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Samples a random batch of experiences from the buffer.
        Returns batches of states, actions, rewards, next_states, and done flags.
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        state = torch.cat(state)
        next_state = torch.cat(next_state)
        action = torch.tensor(action, dtype=torch.long).unsqueeze(1).to(device)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(1).to(device)
        done = torch.tensor(done, dtype=torch.bool).unsqueeze(1).to(device)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """
    Deep Q-Network Agent that interacts with the environment,
    stores experiences, and learns from them.
    """
    def __init__(self, gamma=0.99, lr=0.001, batch_size=64, max_mem=10000,
                 eps_start=1.0, eps_end=0.01, eps_decay=10000, target_update=1000):
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update

        self.steps_done = 0
        self.memory = ReplayBuffer(capacity=max_mem)

        self.policy_net = DuelingDQN().to(device)
        self.target_net = DuelingDQN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss()

    def select_action(self, state):
        """
        Selects an action using epsilon-greedy policy.
        Converts state to Tensor if it's a NumPy array.
        """
        # Ensure state is a Tensor
        if isinstance(state, np.ndarray):
            state = state_to_tensor(state)
            print(f"[DQNAgent] Converted state from NumPy to Tensor.")
        elif isinstance(state, torch.Tensor):
            state = state.to(device)
            print(f"[DQNAgent] Received state as Tensor on device {device}.")
        else:
            raise ValueError("State must be a numpy array or a torch.Tensor")

        eps_threshold = max(self.eps_end, self.eps_start - self.steps_done / self.eps_decay)
        self.steps_done += 1

        if random.random() < eps_threshold:
            action = random.randint(0, len(ACTIONS) - 1)
            print(f"[DQNAgent] Chose random action: {action}")
            return action
        else:
            with torch.no_grad():
                q_values = self.policy_net(state)
                action = q_values.argmax(dim=1).item()
                print(f"[DQNAgent] Chose action based on policy_net: {action}")
                return action

    def store_experience(self, state, action, reward, next_state, done):
        """
        Stores the experience in replay buffer.
        """
        self.memory.push(state, action, reward, next_state, done)
        print(f"[DQNAgent] Stored experience in replay buffer.")

    def update(self):
        """
        Updates the policy network based on sampled experiences.
        """
        if len(self.memory) < self.batch_size:
            print(f"[DQNAgent] Not enough samples to update. Current memory size: {len(self.memory)}")
            return
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)
        print(f"[DQNAgent] Sampled a batch of size {self.batch_size}")

        # Compute current Q values
        q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
            target_q_values = reward_batch + (self.gamma * next_q_values * (~done_batch))

        # Compute loss
        loss = self.loss_fn(q_values, target_q_values)
        print(f"[DQNAgent] Computed loss: {loss.item()}")

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        print(f"[DQNAgent] Optimized the model.")

    def update_target(self):
        """
        Updates the target network with the policy network's weights.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print(f"[DQNAgent] Updated target network.")

def train_agent(env, agent, num_episodes=1000, target_update_freq=100):
    """
    Trains the DQN agent within the given environment for a specified number of episodes.
    Periodically updates the target network.
    """
    for episode in range(num_episodes):
        state = env.reset()
        state = state_to_tensor(state)  # Convert to Tensor
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(ACTIONS[action])
            next_state = state_to_tensor(next_state)  # Convert to Tensor
            agent.store_experience(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            total_reward += reward

        # Update the target network periodically
        if episode % target_update_freq == 0:
            agent.update_target()

        print(f"Episode {episode}, Total Reward: {total_reward}")

# agent/dqn_agent.py
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

ACTIONS = ['up', 'down', 'left', 'right']

def state_to_tensor(board):
    transformed = np.log2(board + 1)
    return torch.tensor(transformed, dtype=torch.float32).view(1, -1)  # (1,16)

class DQN(nn.Module):
    def __init__(self, input_size=16, hidden_size=128, output_size=4):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
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

        self.policy_net = DQN()
        self.target_net = DQN()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1.0 * self.steps_done / self.eps_decay)
        self.steps_done += 1

        if sample < eps_threshold:
            return random.randint(0, 3)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state_to_tensor(state))
                return q_values.argmax(dim=1).item()

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)

        state_batch = torch.cat([state_to_tensor(s) for s in state_batch])
        action_batch = torch.tensor(action_batch, dtype=torch.long).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32).unsqueeze(1)
        next_state_batch = torch.cat([state_to_tensor(s) for s in next_state_batch])
        done_batch = torch.tensor(done_batch, dtype=torch.bool).unsqueeze(1)

        q_values = self.policy_net(state_batch).gather(1, action_batch)
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
        target_q_values = reward_batch + (self.gamma * next_q_values * (~done_batch))

        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

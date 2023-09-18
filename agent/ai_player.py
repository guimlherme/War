import random

import torch.nn.functional as F
from agent.state_action_space import action_space, len_state_space

import torch
import torch.nn as nn
import torch.optim as optim

# Set constants
INTERMEDIATE_LAYER_SIZE = 128
LEARNING_RATE = 1e-6
MEMORY_CAPACITY = 1000

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, valid_actions, action, reward, next_state, done, player_id):
        data = (state, valid_actions, action, reward, next_state, done, player_id)
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.position] = data
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def reset(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


class DQNModel(nn.Module):
    def __init__(self, input_size, num_actions):
        super(DQNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, INTERMEDIATE_LAYER_SIZE)
        self.fc2 = nn.Linear(INTERMEDIATE_LAYER_SIZE, INTERMEDIATE_LAYER_SIZE)
        self.q_values = nn.Linear(INTERMEDIATE_LAYER_SIZE, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q_values(x)


class AIPlayer:
    def __init__(self, name='ai', load_path=None):
        # NN size adjustment
        input_size = len_state_space
        num_actions = len(action_space)

        # Initialize DQN and Target Network

        self.name = name
        self.type = 'dqn_agent'
        self.dqn_model = DQNModel(input_size, num_actions)
        if load_path:
            self.dqn_model.load_state_dict(torch.load(load_path))
        self.target_model = DQNModel(input_size, num_actions)
        self.target_model.load_state_dict(self.dqn_model.state_dict())

        self.optimizer = optim.AdamW(self.dqn_model.parameters(), lr=LEARNING_RATE)
        self.replay_buffer = ReplayBuffer(MEMORY_CAPACITY)
        self.loss = nn.SmoothL1Loss()
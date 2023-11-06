from agent.replay_buffer import ReplayBuffer
import torch.nn.functional as F
from agent.environment import WarEnvironment
from agent.state_action_space import action_space, len_state_space

import torch
import torch.nn as nn
import torch.optim as optim

import os
import random

# Set constants
GAMMA = 0.999 # War is a strategic game, so we need to go for late rewards
INTERMEDIATE_LAYER_SIZE = 1024
LEARNING_RATE = 1e-1
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
TARGET_UPDATE_FREQUENCY = 1000

def validate_q_values(valid_actions_table: list, q_values: torch.Tensor) -> torch.Tensor:
    valid_actions_table = torch.tensor(valid_actions_table, dtype=torch.bool).unsqueeze(0)
    invalid_mask = ~valid_actions_table
    q_values[invalid_mask] = -float('inf')
    return q_values

def select_valid_action(valid_actions_table: list, q_values: torch.Tensor):
    q_values_validated = validate_q_values(valid_actions_table, q_values)
    return torch.argmax(q_values_validated)

class DQNModel(nn.Module):
    def __init__(self, input_size, num_actions):
        super(DQNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, INTERMEDIATE_LAYER_SIZE)
        self.fc2 = nn.Linear(INTERMEDIATE_LAYER_SIZE, INTERMEDIATE_LAYER_SIZE)
        self.fc3 = nn.Linear(INTERMEDIATE_LAYER_SIZE, INTERMEDIATE_LAYER_SIZE)
        self.q_values = nn.Linear(INTERMEDIATE_LAYER_SIZE, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.q_values(x)


class DQNPlayer:
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
        self.loss = nn.MSELoss()

        self.last_target_update = 0

    def select_action(self, state: torch.Tensor, valid_actions_table: list):

        with torch.no_grad():
            q_values = self.dqn_model(state)
        action = select_valid_action(valid_actions_table, q_values)

        if random.random() < 0.001: 
            q_values = validate_q_values(valid_actions_table, q_values)
            print(self.name, torch.max(q_values).item(), action)
            

        
        return action
    
    def update(self):
        # Train the DQN model
        if len(self.replay_buffer) >= BATCH_SIZE:
            batch = self.replay_buffer.sample(BATCH_SIZE)
            states, valid_actions, actions, rewards, next_states, dones, player_indexes = zip(*batch)

            states = torch.cat(states, dim=0)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
            dones_tensor = torch.tensor(dones, dtype=torch.bool)
            actions_tensor = torch.tensor(actions, dtype=torch.int64).unsqueeze(0)
            next_states = torch.cat(next_states, dim=0)

            with torch.no_grad():
                q_values_next = self.target_model(next_states)

            # Initialize target Q-values with rewards
            target_q_values = rewards_tensor.clone()

            # Mask out terminal states (where dones is True)
            non_terminal_states = ~dones_tensor

            max_q_values_next = torch.zeros(BATCH_SIZE, dtype=torch.float32)
            # Compute the maximum Q-value for each batch element while ignoring invalid actions
            for i in range(len(valid_actions)): # Probably it's bettter to keep it in a loop #TODO: test
                max_q_values_next[i] = torch.max(q_values_next[i][valid_actions[i]])

            # Update target Q-values for non-terminal states
            target_q_values[non_terminal_states] += GAMMA * max_q_values_next[non_terminal_states]

            q_values = self.dqn_model(states)

            q_values_actions = q_values.gather(1, actions_tensor).squeeze()

            loss = self.loss(q_values_actions, target_q_values)


            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update Target Network
            self.last_target_update += 1
            if self.last_target_update % TARGET_UPDATE_FREQUENCY == 0:
                self.target_model.load_state_dict(self.dqn_model.state_dict())
            
    def save(self, model_checkpoint_folder, episode):

        if not os.path.exists(model_checkpoint_folder):
            os.makedirs(model_checkpoint_folder)
        
        model_checkpoint_path = os.path.join(model_checkpoint_folder, f"dqn_model_{self.name}_episode_{episode}.pth")
        torch.save(self.dqn_model.state_dict(), model_checkpoint_path)
    
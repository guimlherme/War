import numpy as np
import random
import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agent.environment import WarEnvironment
from agent.state_action_space import len_state_space, action_space
from agent.logger import CustomLogger

# Set constants
INTERMEDIATE_LAYER_SIZE = 128
BATCH_SIZE = 64
GAMMA = 0.999 # War is a strategic game, so we need to go for late rewards
LEARNING_RATE = 1e-5
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQUENCY = 11 # Must be odd to save both models
MEMORY_CAPACITY = 1000
EPISODES = 1000
SAVE_MODEL_FREQUENCY = 5

device = ("cpu")
# Uncomment next line to activate GPU support
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

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

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done, player_id):
        data = (state, action, reward, next_state, done, player_id)
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.position] = data
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class AIPlayer:
    def __init__(self, name='ai'):
        # NN size adjustment
        input_size = len_state_space
        num_actions = len(action_space)

        # Initialize DQN and Target Network

        self.name = name
        self.dqn_model = DQNModel(input_size, num_actions)
        self.target_model = DQNModel(input_size, num_actions)
        self.target_model.load_state_dict(self.dqn_model.state_dict())

        self.optimizer = optim.AdamW(self.dqn_model.parameters(), lr=LEARNING_RATE)
        self.replay_buffer = ReplayBuffer(MEMORY_CAPACITY)
        self.loss = nn.SmoothL1Loss()

def select_valid_action(env: WarEnvironment, q_values: torch.Tensor):
    q_values_numpy = q_values.cpu().squeeze().numpy()
    valid_actions_table = np.array(env.get_valid_actions_table())
    validity_factor = np.where(valid_actions_table, 0, -np.inf)
    q_values_validated = q_values_numpy + validity_factor
    return np.argmax(q_values_validated)


def dqn_learning(env: WarEnvironment, player0 = AIPlayer(name='ai0'), player1 = AIPlayer(name='ai1')):

    model_checkpoint_folder = 'models'
    logger = CustomLogger(log_file="training_log.log")

    num_actions = len(action_space)
    epsilon = EPSILON_START
    for episode in range(EPISODES):
        
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        done = False
        total_reward = 0
        current_training = player0 # Train only one agent
        # current_training = player0 if episode % 2 == 0 else player1 
        current_player = player0

        while not done:
            # Epsilon-Greedy Exploration
            if random.random() < epsilon:
                valid_actions = env.get_valid_actions_indexes()
                action = random.choice(valid_actions)
            else:
                with torch.no_grad():
                    q_values = current_player.dqn_model(state)
                action = select_valid_action(env, q_values)
                if env.game.match_action_counter % 100 == 0: print(current_player.name, 
                                                                   torch.median(q_values).item(), action)
                # action = torch.argmax(q_values).item()

            next_player_index, next_state, next_player_reward = env.step(action_space[action])

            if next_player_index == None:
                done = True
                next_player = current_player
            else:
                next_player = player0 if next_player_index == 0 else player1

            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            reward = next_player_reward
            
            # print(f"Action: phase: {state[0][0]}, player: {current_player}, target: {action_space[action]}, reward: {reward} ")
            # Add experience to the agent's replay buffer
            if next_player == current_training:
                total_reward += reward
                next_player.replay_buffer.add(state, action, reward, next_state, done, current_player)

            state = next_state
            current_player = next_player

            # Train the DQN model
            if len(current_training.replay_buffer) >= BATCH_SIZE:
                batch = current_training.replay_buffer.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones, player_ids = zip(*batch)

                states = torch.tensor(np.concatenate([s.cpu() for s in states]), 
                                      dtype=torch.float32).to(device)
                actions_tensor = torch.tensor(actions, dtype=torch.int64).unsqueeze(0)
                next_states = torch.tensor(np.concatenate([n.cpu() for n in next_states]), 
                                           dtype=torch.float32).to(device)

                q_values_next = current_training.target_model(next_states)

                # Get the target Q-values based on the rewards and next states
                target_q_values = torch.zeros(BATCH_SIZE, dtype=torch.float32)
                for i in range(BATCH_SIZE):
                    if dones[i]:
                        target_q_values[i] = rewards[i]
                    else:
                        target_q_values[i] = rewards[i] + GAMMA * torch.max(q_values_next[i])

                q_values = current_training.dqn_model(states)
                # q_values_actions = torch.sum(torch.nn.functional.one_hot(torch.tensor(actions), num_actions) * q_values, dim=1)
                q_values_actions = q_values.gather(1, actions_tensor).squeeze()
                # print(q_values_actions == q_values_actions_2, q_values_actions, q_values_actions_2)
                loss = current_training.loss(q_values_actions, target_q_values)

                current_training.optimizer.zero_grad()
                loss.backward()
                current_training.optimizer.step()

            # Update Target Network
            if episode % TARGET_UPDATE_FREQUENCY == 0:
                # TODO: update all networks
                current_training.target_model.load_state_dict(current_training.dqn_model.state_dict())

        # Decay Epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        total_reward_msg = f"Episode {episode}, Agent {current_training.name}, Total Reward: {total_reward}"
        logger.info(total_reward_msg)

        # Save the DQN models after some episodes
        if episode % SAVE_MODEL_FREQUENCY == 0:
            if not os.path.exists(model_checkpoint_folder):
                os.makedirs(model_checkpoint_folder)
            
            # Save DQN Model 1
            model_checkpoint_path1 = os.path.join(model_checkpoint_folder, f"dqn_model0_episode_{episode}.pth")
            torch.save(player0.dqn_model.state_dict(), model_checkpoint_path1)

            # Save DQN Model 2
            model_checkpoint_path2 = os.path.join(model_checkpoint_folder, f"dqn_model1_episode_{episode}.pth")
            torch.save(player1.dqn_model.state_dict(), model_checkpoint_path2)


if __name__ == "__main__":
    # Instatiate WarEnvironment with 2 players
    env = WarEnvironment(2)

    dqn_learning(env)

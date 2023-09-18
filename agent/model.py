import numpy as np
import random
import os
import sys
import shutil
import logging
import cProfile
from typing import List, Union

import torch
from agent.ai_player import AIPlayer, VICTORY_REWARD

from agent.environment import WarEnvironment
from agent.logger import CustomLogger
from agent.random_player import RandomPlayer

BATCH_SIZE = 32
GAMMA = 0.999 # War is a strategic game, so we need to go for late rewards
EPSILON_START = 1.0
EPSILON_END = 1.0
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQUENCY = 5 # Must be odd to save both models
EPISODES = 10000
SAVE_MODEL_FREQUENCY = 5

device = ("cpu")
# Uncomment next line to activate GPU support
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

def validate_q_values(env: WarEnvironment, q_values: torch.Tensor) -> torch.Tensor:
    valid_actions_table = torch.tensor(env.get_valid_actions_table(), dtype=torch.bool).unsqueeze(0)
    invalid_mask = ~valid_actions_table
    q_values[invalid_mask] = -float('inf')
    return q_values

def select_valid_action(env: WarEnvironment, q_values: torch.Tensor):
    q_values_validated = validate_q_values(env, q_values)
    return torch.argmax(q_values_validated)

def dqn_learning(env: WarEnvironment, players: List[Union[AIPlayer, RandomPlayer]], start_episode=0):
    
    # Define the number of players
    num_players = len(players)

    # Move models to the device if they are AIPlayers
    for player in players:
        print(f'Player {player.type}: ', type(player))
        if isinstance(player, AIPlayer):
            player.replay_buffer.reset()
            player.dqn_model.to(device)

    model_checkpoint_folder = 'models'
    logger = CustomLogger(log_file="training_log.log")

    epsilon = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** start_episode))
    for episode in range(start_episode, EPISODES):
        
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        done = False
        total_reward = 0
        if not isinstance(players[1], AIPlayer):
            current_training = players[0] # Train only one agent
        else:
            current_training = players[0] if episode % 20 < 10 else players[1] # Change training agent every 10 rounds
        
        starting_player = random.randrange(num_players)
        current_player = players[starting_player]

        while not done:
            # Epsilon-Greedy Exploration
            if random.random() < epsilon or isinstance(current_player, RandomPlayer):
                valid_actions = env.get_valid_actions_indexes()
                action = random.choice(valid_actions)
            else:
                valid_actions = env.get_valid_actions_indexes()
                with torch.no_grad():
                    q_values = current_player.dqn_model(state)
                action = select_valid_action(env, q_values)
                if env.game.match_action_counter % 200 == 0: 
                    q_values = validate_q_values(env, q_values)
                    print(current_player.name, torch.max(q_values).item(), action)
                # action = torch.argmax(q_values).item()

            next_player_index, next_state, next_player_reward = env.step(action)

            if next_player_index == None:
                done = True
                next_player = current_player
            else:
                next_player = players[next_player_index]

            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            reward = next_player_reward
            
            # print(f"Action: phase: {state[0][0]}, player: {current_player}, target: {action_space[action]}, reward: {reward} ")
            # Add experience to the agent's replay buffer
            if next_player == current_training:
                total_reward += reward
                next_player.replay_buffer.add(state, valid_actions, action, reward, next_state, done, current_player)
            elif done and env.ended_in_objective():
                # TODO: think about a better logic for this
                total_reward -= VICTORY_REWARD
                current_training.replay_buffer.register_loss()

            state = next_state
            current_player = next_player

            # Train the DQN model
            if len(current_training.replay_buffer) >= BATCH_SIZE:
                batch = current_training.replay_buffer.sample(BATCH_SIZE)
                states, valid_actions, actions, rewards, next_states, dones, player_ids = zip(*batch)

                states = torch.cat(states, dim=0)
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
                dones_tensor = torch.tensor(dones, dtype=torch.bool)
                actions_tensor = torch.tensor(actions, dtype=torch.int64).unsqueeze(0)
                next_states = torch.cat(next_states, dim=0)

                with torch.no_grad():
                    q_values_next = current_training.target_model(next_states)

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

                q_values = current_training.dqn_model(states)

                q_values_actions = q_values.gather(1, actions_tensor).squeeze()

                loss = current_training.loss(q_values_actions, target_q_values)


                current_training.optimizer.zero_grad()
                loss.backward()
                current_training.optimizer.step()

                # if env.game.match_action_counter % 500 == 0:
                #     with torch.no_grad():
                #         new_q_values = current_training.dqn_model(states)
                #         new_q_values_actions = new_q_values.gather(1, actions_tensor).squeeze()
                #         new_loss = current_training.loss(new_q_values_actions, target_q_values)
                #         print('new_q_values: ' + str(new_q_values))
                #         print('target: ' + str(target_q_values))
                #         print('loss: ' + str(loss))
                #         print('new_loss: ' + str(new_loss))

            # Update Target Network
            if episode % TARGET_UPDATE_FREQUENCY == 0:
                # TODO: maybe update all networks
                current_training.target_model.load_state_dict(current_training.dqn_model.state_dict())

        # Decay Epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        total_reward_msg = (f"Episode {episode}, Agent {current_training.name}, Total Reward: {total_reward}, " + 
                            f"Num actions: {env.get_match_action_counter()}, Epsilon: {epsilon}")
        logger.info(total_reward_msg)

        # Save the DQN models after some episodes
        if episode % SAVE_MODEL_FREQUENCY == 0:
            if not os.path.exists(model_checkpoint_folder):
                os.makedirs(model_checkpoint_folder)
            
            # Save DQN Model 1
            if players[0].type != 'random_agent':
                model_checkpoint_path1 = os.path.join(model_checkpoint_folder, f"dqn_model0_episode_{episode}.pth")
                torch.save(players[0].dqn_model.state_dict(), model_checkpoint_path1)

            # Save DQN Model 2
            if players[1].type != 'random_agent':
                model_checkpoint_path2 = os.path.join(model_checkpoint_folder, f"dqn_model1_episode_{episode}.pth")
                torch.save(players[1].dqn_model.state_dict(), model_checkpoint_path2)

def main():
    # Verify if the user demanded to train with a random agent or itself
    # No need to use parser just for this
    force_random_agent = (len(sys.argv) >= 2 and sys.argv[1] == '-r')
    force_train_itself = (len(sys.argv) >= 2 and sys.argv[1] == '-d')

    # Instantiate WarEnvironment with the desired number of players
    num_players = 2 if len(sys.argv) <= 2 else int(sys.argv[2])
    env = WarEnvironment(num_players)

    # If resuming the training:
    try:
        checkpoint_files = os.listdir("models")
        episodes = [int(f.split('.')[0].split('_')[3]) for f in checkpoint_files]
    except FileNotFoundError:
        episodes = None
    
    if episodes:
        episode_checkpoint = max(episodes)
        players = []
        
        for i in range(num_players):
            if i == 0:
                players.append(AIPlayer(name='ai_0', load_path=f"models/dqn_model0_episode_{episode_checkpoint}.pth"))
            if (i == 1 and (os.path.exists(f"models/dqn_model1_episode_{episode_checkpoint}.pth") 
                            and not force_random_agent)):
                players.append(AIPlayer(name='ai_1', load_path=f"models/dqn_model1_episode_{episode_checkpoint}.pth"))
            elif i == 1 and force_train_itself:
                players.append(AIPlayer(name='ai_1', load_path=f"models/dqn_model0_episode_{episode_checkpoint}.pth"))
            else:
                players.append(RandomPlayer(name=f'ai_{i}'))

        dqn_learning(env, players, start_episode=episode_checkpoint)
    
    else:
        episode_checkpoint = 0
        players = []
        
        for i in range(num_players):
            if i==0:
                players.append(AIPlayer(name='ai_0'))
            if i==1 and force_train_itself:
                players.append(AIPlayer(name='ai_1'))
            else:
                players.append(RandomPlayer(name=f'ai_{i}'))

        dqn_learning(env, players, start_episode=episode_checkpoint)

if __name__ == "__main__":
    main()
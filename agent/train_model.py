import numpy as np
import random
import os
import sys
import shutil
import logging
import cProfile
from typing import List, Union

import torch

from agent.environment import WarEnvironment
from agent.logger import CustomLogger
from agent.random_player import RandomPlayer
from agent.ddqn.ai_player import DQNPlayer
from game.war import VICTORY_REWARD

EPSILON_START = 1.0
EPSILON_END = 1e-3
EPSILON_DECAY = 0.995
EPISODES = 10000
SAVE_MODEL_FREQUENCY = 5

device = ("cpu")
# Uncomment next line to activate GPU support
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)


def dqn_learning(env: WarEnvironment, players: List[Union[DQNPlayer, RandomPlayer]], start_episode=0):
    
    # Define the number of players
    num_players = len(players)

    # Move models to the device if they are AIPlayers
    for player in players:
        print(f'Player {player.type}: ', type(player))
        if not isinstance(player, RandomPlayer):
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
        if isinstance(players[1], RandomPlayer):
            current_training = players[0] # Train only one agent
        else:
            current_training = players[0] if episode % 20 < 10 else players[1] # Change training agent every 10 rounds
        
        current_training_index = players.index(current_training)
        starting_player = random.randrange(num_players)
        current_player = players[starting_player]
        current_training_last_state = None
        current_training_last_action = None
        current_training_last_valid_actions = None

        while not done:

            valid_actions_table, valid_actions = env.get_valid_actions_table_and_indexes()

            # Epsilon-Greedy Exploration
            if random.random() < epsilon or isinstance(current_player, RandomPlayer):
                action = random.choice(valid_actions)
            else:
                action = current_player.select_action(state, valid_actions_table)

            next_player_index, next_state, next_player_reward = env.step(action)

            if next_player_index == None:
                done = True
                next_player = current_player
            else:
                next_player = players[next_player_index]

            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            reward = next_player_reward
            state = next_state
            current_player = next_player
            
            # Add experience to the agent's replay buffer
            if next_player == current_training:
                if current_training_last_state is not None:
                    total_reward += reward
                    assert current_training_last_state[0][1] == next_state[0][1] # Objective shouldn't change
                    next_player.replay_buffer.add(current_training_last_state, 
                                                  current_training_last_valid_actions, 
                                                  current_training_last_action, 
                                                  reward, 
                                                  next_state, 
                                                  done, 
                                                  current_training_index)
                current_training_last_state = next_state
                current_training_last_action = action
                current_training_last_valid_actions = valid_actions
            elif done and (env.ended_in_objective() or env.player_has_died(current_training_index)):
                # TODO: think about a better logic for this
                total_reward -= VICTORY_REWARD
                current_training.replay_buffer.register_loss()
            elif done:
                current_training.replay_buffer.register_end()

            # Train the model
            current_player.update()
            

        # Decay Epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        total_reward_msg = (f"Episode {episode}, Agent {current_training.name}, Total Reward: {total_reward:.2f}, " + 
                            f"Num actions: {env.get_match_action_counter()}, Epsilon: {epsilon:.2f}")
        logger.info(total_reward_msg)

        # Save the models after some episodes
        if episode % SAVE_MODEL_FREQUENCY == 0:
            for player in players:
                player.save(model_checkpoint_folder, episode)

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
                players.append(DQNPlayer(name='ai_0', load_path=f"models/dqn_model0_episode_{episode_checkpoint}.pth"))
            elif (i == 1 and (os.path.exists(f"models/dqn_model1_episode_{episode_checkpoint}.pth") 
                            and not force_random_agent)):
                players.append(DQNPlayer(name='ai_1', load_path=f"models/dqn_model1_episode_{episode_checkpoint}.pth"))
            elif i == 1 and force_train_itself:
                players.append(DQNPlayer(name='ai_1', load_path=f"models/dqn_model0_episode_{episode_checkpoint}.pth"))
            else:
                players.append(RandomPlayer(name=f'ai_{i}'))

        dqn_learning(env, players, start_episode=(episode_checkpoint+1))
    
    else:
        episode_checkpoint = 0
        players = []
        
        for i in range(num_players):
            if i==0:
                players.append(DQNPlayer(name='ai_0'))
            elif i==1 and force_train_itself:
                players.append(DQNPlayer(name='ai_1'))
            else:
                players.append(RandomPlayer(name=f'ai_{i}'))

        dqn_learning(env, players, start_episode=(episode_checkpoint+1))

if __name__ == "__main__":
    main()

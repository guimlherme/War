import os
from agent.ddqn.ai_player import DQNPlayer
from agent.random_player import RandomPlayer

from agent.environment import WarEnvironment
from agent.train_model import select_valid_action
from agent.state_action_space import len_state_space, action_space

import torch

EPISODES = 10

def test_model(env, players):
    for episode in range(EPISODES):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        done = False
        total_reward_0 = 0
        total_reward_1 = 0
        current_player_index = 0
        current_player = players[current_player_index]
        

        while not done:
            # Greedy Evaluation
            with torch.no_grad():
                q_values = current_player.dqn_model(state)
                action = select_valid_action(env, q_values)

            next_player_index, next_state, next_player_reward = env.step(action)
            # print(next_player_reward)

            if next_player_index == None:
                done = True
                next_player = current_player
            else:
                current_player_index = (current_player_index + 1) % len(players)
                next_player = players[current_player_index]

            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            reward = next_player_reward
            
            # print(f"Action: phase: {state[0][0]}, player: {current_player}, target: {action_space[action]}, reward: {reward} ")
            # Add experience to the agent's replay buffer
            if next_player == players[0]:
                total_reward_0 += reward
            elif next_player == players[1]:
                total_reward_1 += reward

            state = next_state
            current_player = next_player

        territories_owners = state[0][2::2] #doesnt work
        # current_player_numbers = [0] if current_player==players[0] else [1,2]
        # other_player_numbers = [1,2] if current_player==players[0] else [0]

        # player0_territories = len([t for t in territories_owners if t in current_player_numbers])
        # player1_territories = len([t for t in territories_owners if t in other_player_numbers])
        # print(state, player0_territories, player1_territories)
        
        total_reward_msg_0 = f"Episode {episode}, Agent {players[0].name}, Total Reward: {total_reward_0}"
        total_reward_msg_1 = f"Episode {episode}, Agent {players[1].name}, Total Reward: {total_reward_1}"

        print(total_reward_0)
        print(total_reward_1)

if __name__ == "__main__":

    # # Define loaded model path
    # model0_episode = 5
    # model1_episode = 1455

    # base_path = os.getcwd()

    # model0_path = os.path.join(base_path, f'models/dqn_model0_episode_{model0_episode}.pth')
    # model1_path = os.path.join(base_path, f'models/dqn_model0_episode_{model1_episode}.pth')

    # Instatiate WarEnvironment with 5 players
    env = WarEnvironment(5)

    # # Instantiate AI players and load trained parameters
    # player0 = DQNPlayer('ai_0')
    # if os.path.exists(model1_path):
    #     player1 = DQNPlayer('ai1')
    # else:
    #     player1 = RandomPlayer('ai_1')


    # player0.dqn_model.load_state_dict(torch.load(model0_path))
    # player1.dqn_model.load_state_dict(torch.load(model1_path))

    player0 = MCTSPlayer('ai_0')
    player1 = RandomPlayer('ai_1')
    player2 = RandomPlayer('ai_2')
    player3 = RandomPlayer('ai_3')
    player4 = RandomPlayer('ai_4')
    players = [player0, player1, player2, player3, player4]
    test_model(env, players)

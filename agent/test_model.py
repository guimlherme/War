import os
from agent.ai_player import AIPlayer

from agent.environment import WarEnvironment
from agent.model import select_valid_action
from agent.state_action_space import len_state_space, action_space

import torch

EPISODES = 10

def test_model(env, player0, player1):
    for episode in range(EPISODES):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        done = False
        total_reward_0 = 0
        total_reward_1 = 0
        current_player = player0

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
                next_player = player0 if next_player_index == 0 else player1

            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            reward = next_player_reward
            
            # print(f"Action: phase: {state[0][0]}, player: {current_player}, target: {action_space[action]}, reward: {reward} ")
            # Add experience to the agent's replay buffer
            if next_player == player0:
                total_reward_0 += reward
            else:
                total_reward_1 += reward

            state = next_state
            current_player = next_player

        territories_owners = state[0][1::2]
        current_player_numbers = [0] if current_player==player0 else [1,2]
        other_player_numbers = [1,2] if current_player==player0 else [0]

        player0_territories = len([t for t in territories_owners if t in current_player_numbers])
        player1_territories = len([t for t in territories_owners if t in other_player_numbers])
        print(state, player0_territories, player1_territories)
        
        total_reward_msg_0 = f"Episode {episode}, Agent {player0.name}, Total Reward: {total_reward_0}"
        total_reward_msg_1 = f"Episode {episode}, Agent {player1.name}, Total Reward: {total_reward_1}"

        print(total_reward_0)
        print(total_reward_1)

if __name__ == "__main__":

    # Define loaded model path
    model0_episode = 1855
    model1_episode = 1855

    base_path = os.getcwd()

    model0_path = os.path.join(base_path, f'models/dqn_model0_episode_{model0_episode}.pth')
    model1_path = os.path.join(base_path, f'models/dqn_model0_episode_{model1_episode}.pth')

    # Instatiate WarEnvironment with 2 players
    env = WarEnvironment(2)

    # Instantiate AI players and load trained parameters
    player0 = AIPlayer('ai0')
    player1 = AIPlayer('ai1')

    player0.dqn_model.load_state_dict(torch.load(model0_path))
    player1.dqn_model.load_state_dict(torch.load(model1_path))
    test_model(env, player0, player1)

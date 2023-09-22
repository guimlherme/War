from agent.environment import WarEnvironment

import torch
import random

class RandomPlayer:
    def __init__(self, name='random_ai'):
        self.name = name
        self.type = 'random_agent'

    def select_action(state: torch.Tensor, valid_actions_table: list):
        valid_actions = [i for i in range(len(valid_actions_table)) if valid_actions_table[i] == True]
        return random.choice(valid_actions)
    
    def update(self):
        pass

    def save(self, model_checkpoint_folder, episode):
        pass
    

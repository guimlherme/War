from game.war import VICTORY_REWARD

import random


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, valid_actions, action, reward, next_state, done, player_index):
        # If you change this, remember to revise all functions in this class
        data = (state, valid_actions, action, reward, next_state, done, player_index)
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.position] = data
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def register_loss(self):
        new_data = list(self.buffer[-1])
        new_data[3] -= VICTORY_REWARD
        new_data[5] = True
        self.buffer[-1] = tuple(new_data)

    def register_end(self):
        new_data = list(self.buffer[-1])
        new_data[5] = True
        self.buffer[-1] = tuple(new_data)

    def reset(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)
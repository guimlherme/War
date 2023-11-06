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
        last_position = (self.position - 1) % self.capacity
        new_data = list(self.buffer[last_position])
        new_data[3] -= VICTORY_REWARD
        new_data[5] = True
        self.buffer[last_position] = tuple(new_data)
        # for _ in range(200):
        #     self.add(*new_data)

    def register_end(self):
        last_position = (self.position - 1) % self.capacity
        new_data = list(self.buffer[last_position])
        new_data[5] = True
        self.buffer[last_position] = tuple(new_data)
        # for _ in range(200):
        #     self.add(*new_data)

    def reset(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)
from game.war import Game
from agent.state_action_space import action_space


class WarEnvironment:
    def __init__(self, num_players):
        self.num_players = num_players
        self.game = Game(self.num_players, debug=False, objectives_enabled=True)

    def reset(self):
        return self.game.reset()

    def step(self, action, current_player_index=None):
        return self.game.play_round(action=action_space[action])
    
    # def get_valid_actions_from_state(self, state):
    #     return self.game.get_valid_actions_from_state(state)

    def get_valid_actions_table(self):
        return self.game.get_valid_actions_table()
    
    def get_valid_actions_indexes(self):
        valid_actions_table = self.get_valid_actions_table()
        return [i for i in range(len(valid_actions_table)) if valid_actions_table[i] == True]
    
    def get_match_action_counter(self):
        return self.game.match_action_counter

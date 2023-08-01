from game.war import Game


class WarEnvironment:
    def __init__(self, num_players):
        self.num_players = num_players
        # TODO: train with objectives
        self.game = Game(self.num_players, debug=False, objectives_enabled=False)

    def reset(self):
        return self.game.reset()

    def step(self, action):
        return self.game.play_round(action=action)
        

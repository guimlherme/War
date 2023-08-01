"""
State structure:
[
    1 x phase: {
        0 = reinforce
        1 = attack
        2 = transfer
    }
    len(territories) x territories: [
        owner: {0 = self, 1 = enemy}
        number of troops: int
    ]      
]

Action structure:
[
    1 x action: {
        -1 = pass
        country_index: int
    }
    
]
Attacks will run until conquered or can't proceed. Transfers will happen with all troops.
"""

BAD_SELECTION_PENALTY = -0.02

from game.players.player import Player
from game.utils import debug_print

import random

class AIPlayer(Player):
    def __init__(self, name, color, objective):
        super().__init__(name, color, objective)
        self.is_human = False
        self.action = None # Remember to reset this to None after using

    def get_action(self):
        territory_index = self.action
        self.territory_index = None

        if territory_index == None:
            raise ValueError("Got a null action")

        if territory_index == -1:
            return None
        
        territory = self.board.territories_data[territory_index]

        return territory

    def place_troops(self, troops_to_place, countries = None):

        if countries:
            selected_territory = random.choice(countries)
            selected_territory.troops += num_troops
            self.remaining_troops_to_place = 0
            return

        if not countries:
            countries = self.territories

        selected_territory = self.get_action()

        if selected_territory == None:
            self.reward_parcel += BAD_SELECTION_PENALTY
            return

        num_troops = 1 # Place one by one
        remaining_troops = troops_to_place

        if selected_territory not in self.territories:
            self.reward_parcel += BAD_SELECTION_PENALTY
            return

        if num_troops <= remaining_troops:
            selected_territory.troops += num_troops
            remaining_troops -= num_troops
        else:
            selected_territory.troops += remaining_troops
            remaining_troops -= remaining_troops
        
        self.remaining_troops_to_place = remaining_troops
        return

    def prepare_attack(self, board):

        defender_territory = self.get_action()

        if defender_territory == None:
            return False, False

        if defender_territory.owner == self:
            self.reward_parcel += BAD_SELECTION_PENALTY
            return False, True

        possible_attackers = [t for t in defender_territory.neighbors if t.owner == self and t.troops > 1]

        if len(possible_attackers) == 0:
            self.reward_parcel += BAD_SELECTION_PENALTY
            return False, True
        
        attacker_territory = max(possible_attackers, key=lambda t: t.troops)
        
        return [attacker_territory, defender_territory]
    
    def prepare_transfer(self, board):

        receiver_territory = self.get_action()

        if receiver_territory == None:
            return False
        
        if receiver_territory.owner != self:
            self.reward_parcel += BAD_SELECTION_PENALTY
            return True

        possible_givers = [t for t in receiver_territory.neighbors if t.owner == self and t.troops > 1]

        if len(possible_givers) == 0:
            self.reward_parcel += BAD_SELECTION_PENALTY
            return False
        
        giver_territory = min(possible_givers, key=lambda t: t.troops)
        
        return [giver_territory, receiver_territory]

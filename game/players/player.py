from game.utils import human_selector

class Player:
    def __init__(self, name, color, objective):
        self.name = name
        self.color = color
        self.objective = objective
        self.board = None
        self.territories = []
        self.cards = []
        self.last_territory_len = 0
        self.reward_parcel = 0
        self.has_died = False
        self.has_conquered = False
        self.has_won = False
        self.remaining_troops_to_place = 0

    def set_board(self, board):
        self.board = board

    def players_dict(self, player):
        # Attributes a number to each player based on this player's referential
        # For now, it's only 0=me, 1=others
        return 0 if player==self else 1
    
    def calculate_territory_change(self):
        if self.last_territory_len == 0:
            self.last_territory_len = len(self.territories)
            return 0
        change = len(self.territories) - self.last_territory_len
        self.last_territory_len = len(self.territories)
        return change

    def get_card_types(self):
        return [card.get_card_type() for card in self.cards]
    
    def count_card_types(self):
        counting = {
            "Circle": 0,
            "Square": 0,
            "Triangle": 0
        }
        for card_type in self.get_card_types():
            counting[card_type] += 1
        return counting

    def remove_one_card_from_type(self, type):
        card_types = [card.get_card_type() for card in self.cards]
        index = card_types.index(type)
        card = self.cards[index]
        self.cards.remove(card)
        return card

    def add_territory(self, territory):
        self.territories.append(territory)

    def remove_territory(self, territory):
        self.territories.remove(territory)
        
    def place_troops(*args):
        # Should be implemented in child classes
        raise NotImplementedError
    

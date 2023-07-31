from game.utils import human_selector

class Player:
    def __init__(self, name, color, objective):
        self.name = name
        self.color = color
        self.objective = objective
        self.territories = []
        self.cards = []

    def get_card_types(self):
        return [card.get_card_type() for card in self.cards]

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

    def round_base_placement(self, num_troops):
        self.place_troops(num_troops)

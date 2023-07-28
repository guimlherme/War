from utils import selector

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

    def place_troops(self, num_troops):
        print(f"{self.name}, place {num_troops} troops on your territories.")
        remaining_troops = num_troops

        while remaining_troops > 0:

            selected_territory = selector(self.territories,
                                                    "\nYour territories:",
                                                    "\nEnter the number of the territory to put troops on (0 to finish): ",
                                                    allow_zero=True)
            
            if selected_territory == 0:
                break

            troops_to_place = int(input(f"How many troops to place on {selected_territory.name}? "))

            if troops_to_place <= remaining_troops:
                selected_territory.troops += troops_to_place
                remaining_troops -= troops_to_place
            else:
                print("You don't have enough troops remaining.")
        

    def round_base_placement(self, num_troops):
        self.place_troops(num_troops)

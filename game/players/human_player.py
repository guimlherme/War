from game.players.player import Player
from game.utils import human_selector

import random

class HumanPlayer(Player):
    def __init__(self, name, color, objective):
        super().__init__(name, color, objective)
        self.is_human = True

    def place_troops(self, num_troops, countries = None):
        if not countries:
            countries = self.territoriess

        print(f"{self.name}, place {num_troops} troops on your territories.")
        remaining_troops = num_troops

        while remaining_troops > 0:

            selected_territory = human_selector(countries,
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
                selected_territory.troops += remaining_troops
                remaining_troops -= remaining_troops

from game.players.player import Player
from game.utils import human_selector, debug_print

import random

class HumanPlayer(Player):
    def __init__(self, name, color, objective):
        super().__init__(name, color, objective)
        self.is_human = True

    def get_state():
        return None

    def place_troops(self, num_troops, countries = None):
        if not countries:
            countries = self.territories

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

    def prepare_attack(self, board):
        # Assuming a manual attack; you can add more sophisticated game mechanics later
        attacker_territory = human_selector(self.territories,
                                            "\nYour territories:",
                                            f"\n{self.name}, choose a territory to attack from (0 to cancel): ",
                                            allow_zero=True)
        if attacker_territory == 0:
            return False, False
        attacker_territory = next((t for t in board if t == attacker_territory and t.owner == self), None)

        if not attacker_territory:
            debug_print("\nInvalid territory selection. Try again.")
            return False, True

        defender_territory = human_selector([t for t in attacker_territory.neighbors if t.owner != self],
                                            "\nLinked enemy territories:",
                                            f"\n{self.name}, choose a territory to attack (0 to cancel): ",
                                            allow_zero=True)
        if defender_territory == 0:
            return False, True
        # TODO: improve next line (e.g. in board -> self.territories)
        defender_territory = next((t for t in board if t == defender_territory and t in attacker_territory.neighbors), None)

        if not defender_territory:
            debug_print("Invalid target territory selection. Try again.")
            return False, True
        
        return [attacker_territory, defender_territory]
    
    def prepare_transfer(self, board):
        # Assuming a manual transfer; you can add more sophisticated game mechanics later
        giver_territory = human_selector(self.territories,
                                            "\nYour territories:",
                                            f"\n{self.name}, choose a territory to transfer from (0 to cancel): ",
                                            allow_zero=True)
        if giver_territory == 0:
            return False
        giver_territory = next((t for t in board if t == giver_territory and t.owner == self), None)

        receiver_territory = human_selector([t for t in giver_territory.neighbors if t.owner == self],
                                            "\nLinked territories:",
                                            f"\n{self.name}, choose a territory to transfer to (0 to cancel): ",
                                            allow_zero=True)
        if receiver_territory == 0:
            return True
        receiver_territory = next((t for t in board if t == receiver_territory and t in giver_territory.neighbors), None)

        if not receiver_territory:
            debug_print("Invalid target territory selection. Try again.")
            return True
        
        return [giver_territory, receiver_territory]

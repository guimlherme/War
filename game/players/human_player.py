from game.players.player import Player
from game.utils import human_selector, debug_print

import random

class HumanPlayer(Player):
    def __init__(self, name, color, objective, godmode=False):
        super().__init__(name, color, objective, is_human=True, godmode=godmode)

    def get_state():
        return None

    def place_troops(self, troops_to_place, countries = None):
        if not countries:
            countries = self.territories

        print(f"{self.name}, place {troops_to_place} troops on your territories.")
        remaining_troops = troops_to_place

        while remaining_troops > 0:

            selected_territory = human_selector(countries,
                                          f"\nRemaining troops:{remaining_troops}\nYour territories:",
                                          "\nEnter the number of the territory to put troops on: ",
                                          allow_zero=False)

            troops_to_place = int(input(f"How many troops to place on {selected_territory.name}? "))

            if troops_to_place <= remaining_troops:
                selected_territory.troops += troops_to_place
                remaining_troops -= troops_to_place
            else:
                selected_territory.troops += remaining_troops
                remaining_troops -= remaining_troops
        self.remaining_troops_to_place = 0
        return

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

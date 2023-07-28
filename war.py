import random
from players import Player
from territories import Territory, TerritoryCard, add_links, territories_data
from utils import territory_selector
import numpy as np

def roll_dice():
    return random.randint(1, 6)

class Game:
    def __init__(self, player_names = ['heitor', '01']):
        self.players = [Player(player_name) for player_name in player_names]
        self.current_card_troop_exchange = 4
        self.board = self.setup_board()
        self.territory_cards = self.setup_territory_cards()

    def setup_board(self):
        
        
        # Setting initial territories
        territories = territories_data.copy()
        territories = [Territory(territory_info[0], owner=None, troops=1) for territory_info in territories]

        add_links(territories)  # Add links between territories
        
        random.shuffle(territories)

        board = []
        split_territories = np.array_split(territories, len(self.players))
        for k, player in enumerate(self.players):
            for territory in split_territories[k].tolist():
                territory.owner = player
                player.add_territory(territory)
                board.append(territory)
        
        return board
    
    def setup_territory_cards(self):
        # Create TerritoryCard objects for each country
        territory_cards = [TerritoryCard(territory_name, card_type) 
                           for territory_name, card_type in territories_data]
        return territory_cards

    def display_board(self):
        for territory in self.board:
            print(territory)

    def conquer_territory(self, attacker, defender):
        defender.owner.remove_territory(defender)
        attacker.owner.add_territory(defender)

        defender.owner = attacker.owner

        transfered_troops = input(f"\n{attacker.owner.name}, choose a number of troops to transfer (between 1 and {attacker.troops - 1}): ")
        transfered_troops = int(transfered_troops)
        transfered_troops = min(max(1, transfered_troops), 3)

        # Guarantee that at least 1 troop remains in each territory
        transfered_troops = max(1, attacker.troops - transfered_troops)
        print(f"\nTransfered troops:{transfered_troops}")

        defender.troops = transfered_troops
        attacker.troops = attacker.troops - transfered_troops

    def attack(self, attacker, defender):
        # Simulate dice rolls and determine the outcome of the battle
        attacker_dice = [roll_dice() for _ in range(min(attacker.troops - 1, 3))]
        defender_dice = [roll_dice() for _ in range(min(defender.troops, 3))]

        attacker_dice.sort(reverse=True)
        defender_dice.sort(reverse=True)

        print(f"{attacker.name} rolled: {attacker_dice}")
        print(f"{defender.name} rolled: {defender_dice}")

        # Compare the dice rolls to decide the battle result
        while attacker_dice and defender_dice:
            if attacker_dice[0] > defender_dice[0]:
                defender.troops -= 1
            else:
                attacker.troops -= 1
            attacker_dice.pop(0)
            defender_dice.pop(0)

        if defender.troops <= 0:
            self.conquer_territory(attacker, defender)
            print(f"{attacker.name} won the battle and conquered {defender.name}!")
        else:
            print(f"{defender.name} successfully defended against {attacker.name}'s attack!")

    def get_player_by_name(self, name):
        for player in self.players:
            if player.name == name:
                return player
        return None
    
    # Card methods

    def draw_card(self):
        return self.cards.pop()

    def exchange_cards_for_troops(self, player):
        sets = [
            ["Circle", "Circle", "Circle"],
            ["Square", "Square", "Square"],
            ["Triangle", "Triangle", "Triangle"],
            ["Circle", "Square", "Triangle"]
        ]

        for card_set in sets:
            if all(card in player.get_card_types() for card in card_set):
                for card in card_set:
                    removed_card = player.remove_one_card_from_type(card)
                    self.territory_cards.append(removed_card)
                    random.shuffle(self.territory_cards)
                num_troops = self.current_card_troop_exchange
                self.current_card_troop_exchange += 4
                return num_troops

        return 0

    def reinforce(self, player):
        troops_to_reinforce = self.exchange_cards_for_troops(player)
        if troops_to_reinforce:
            print(f"{player.name} exchanged cards for {troops_to_reinforce} troops.")

        # Placing troops on owned territories
        print(f"\n--- {player.name}'s Reinforcement Phase ---")
        player.place_troops(troops_to_reinforce)


    def start_round(self, player):
        player.round_base_placement(3)
        self.reinforce(player)

    def attack_phase(self, player):
        # Assuming a manual attack; you can add more sophisticated game mechanics later
        attacker_territory = territory_selector(player.territories,
                                                "\nYour territories:",
                                                f"\n{player.name}, choose a territory to attack from (0 to cancel): ",
                                                allow_zero=True)
        if attacker_territory == 0:
            return
        attacker_territory = next((t for t in self.board if t.name == attacker_territory.name and t.owner.name == player.name), None)

        if not attacker_territory:
            print("\nInvalid territory selection. Try again.")
            return

        defender_territory = territory_selector(attacker_territory.neighbors,
                                                "\nLinked territories:",
                                                f"\n{player.name}, choose a territory to attack (0 to cancel): ",
                                                allow_zero=True)
        if defender_territory == 0:
            return
        defender_territory = next((t for t in self.board if t.name == defender_territory.name and t in attacker_territory.neighbors), None)

        if not defender_territory:
            print("Invalid target territory selection. Try again.")
            return

        self.attack(attacker_territory, defender_territory)

    def play(self):
        current_player_index = 0
        while True:
            current_player = self.players[current_player_index]
            self.start_round(current_player)

            print(f"\n--- {current_player.name}'s Turn ---")
            self.display_board()
            
            finished_attacking = 0
            while not finished_attacking:
                self.attack_phase(current_player)
                try:
                    finished_attacking = int(input("\nDo you wish to stop attacking? 0=no, 1=yes\n"))
                except ValueError:
                    print('Invalid Selection, assuming True')
                    finished_attacking = 1

            # Check if the game is over
            # TODO: implement the winning condition cards
            if len(set(t.owner for t in self.board)) == 1:
                print(f"\nCongratulations! {current_player.name} won the game!")
                break

            # Switch to the other player for the next turn
            current_player_index = (current_player_index + 1) % len(self.players)


if __name__ == "__main__":
    players = []
    while True:
        player_name = input("Enter Player's name (or leave blank to start the game): ")
        if not player_name:
            break
        players.append(player_name)
    
    risk_game = Game(players)
    risk_game.play()

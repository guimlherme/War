import os
import random
from players import Player
from territories import Territory, TerritoryCard, territories_data
from utils import selector
from win_conditions import check_win, objectives, objectives_descriptions
import numpy as np

def roll_dice():
    return random.randint(1, 6)

class Game:
    def __init__(self, players):
        self.players = players
        self.current_card_troop_exchange = 4
        self.board = self.setup_board()
        self.cards = self.setup_territory_cards()

    def setup_board(self):

        board = territories_data.copy()

        territories = territories_data.copy()
        random.shuffle(territories)
        split_territories = np.array_split(territories, len(self.players))
        for k, player in enumerate(self.players):
            for territory in split_territories[k].tolist():
                territory.owner = player
                player.add_territory(territory)
        
        return board
    
    def setup_territory_cards(self):
        # Create TerritoryCard objects for each country
        territory_cards = [TerritoryCard(territory.name, territory.shape) 
                           for territory in territories_data]
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

        print(f"{attacker.owner} rolled: {attacker_dice}")
        print(f"{defender.owner} rolled: {defender_dice}")

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
            return True
        else:
            print(f"{defender.name} successfully defended against {attacker.name}'s attack!")
            return False

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
                    self.cards.append(removed_card)
                    random.shuffle(self.cards)
                num_troops = self.current_card_troop_exchange
                self.current_card_troop_exchange += 4
                return num_troops

        return 0

    def reinforce(self, player):
        troops_to_reinforce = self.exchange_cards_for_troops(player)
        if not troops_to_reinforce:
            return
        
        print(f"{player.name} exchanged cards for {troops_to_reinforce} troops.")

        # Placing troops on owned territories
        print(f"\n--- {player.name}'s Reinforcement Phase ---")
        player.place_troops(troops_to_reinforce)


    def start_round(self, player):
        player.round_base_placement(3)

        while True:
            try:
                bool_reinforce = input("Do you want to exchange your cards, if possible? 0=no, 1=yes ")
                if bool_reinforce == '1':
                    self.reinforce(player)
                    break
                elif bool_reinforce == '0':
                    break
                else:
                    raise ValueError
            except ValueError:
                print("\nTry again")

    def attack_phase(self, player):
        # Assuming a manual attack; you can add more sophisticated game mechanics later
        attacker_territory = selector(player.territories,
                                                "\nYour territories:",
                                                f"\n{player.name}, choose a territory to attack from (0 to cancel): ",
                                                allow_zero=True)
        if attacker_territory == 0:
            return False
        attacker_territory = next((t for t in self.board if t == attacker_territory and t.owner == player), None)

        if not attacker_territory:
            print("\nInvalid territory selection. Try again.")
            return False

        defender_territory = selector([t for t in attacker_territory.neighbors if t.owner != player],
                                                "\nLinked territories:",
                                                f"\n{player.name}, choose a territory to attack (0 to cancel): ",
                                                allow_zero=True)
        if defender_territory == 0:
            return False
        defender_territory = next((t for t in self.board if t == defender_territory and t in attacker_territory.neighbors), None)

        if not defender_territory:
            print("Invalid target territory selection. Try again.")
            return False

        return self.attack(attacker_territory, defender_territory)

    def play(self):
        current_player_index = 0
        while True:
            current_player = self.players[current_player_index]
            self.start_round(current_player)

            print(f"\n--- {current_player.name}'s Turn ---")
            self.display_board()
            
            success = False
            finished_attacking = 0
            while not finished_attacking:
                success = self.attack_phase(current_player) or success
                try:
                    finished_attacking = int(input("\nDo you wish to stop attacking? 0=no, 1=yes\n"))
                except ValueError:
                    print('Invalid Selection, assuming True')
                    finished_attacking = 1

            # Give cards if one capture is done
            if success:
                print("Card drawn")
                current_player.cards.append(self.draw_card())
            

            # Check if the game is over
            if check_win(self.players):
                winner = check_win(self.players)
                print(f"\nCongratulations! {winner.name} won the game!")
                break

            # Switch to the other player for the next turn
            current_player_index = (current_player_index + 1) % len(self.players)


if __name__ == "__main__":
    players = []
    colors = ['Azul', 'Amarelo', 'Vermelho', 'Cinza', 'Roxo', 'Verde']
    objectives_list = objectives.copy()
    while True:
        player_name = input("\nEnter Player's name (or leave blank to start the game): ")
        if not player_name:
            break
        color = selector(colors, "\nCores disponíveis:","\nSelecione uma cor para o jogador:", False)
        colors.remove(color)
        objective = random.choice(objectives_list)
        objectives_list.remove(objective)
        input("\nPress enter to see your objective\n")
        print(objectives_descriptions[objective])
        input("\nPress enter to hide your objective\n")
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\n\n\n\n\n\n\n\n\n\n\n")
        players.append(Player(player_name, color, objective))
    
    risk_game = Game(players)
    risk_game.play()

import os
import random
from game.players.human_player import HumanPlayer
from game.players.ai_player import AIPlayer

from game.territories import Territory, TerritoryCard, territories_data, verify_conquered_continents, continent_to_troops
from game.utils import human_selector, debug_print
from game.win_conditions import check_win, simple_check_win, objectives, objectives_descriptions
import numpy as np

REINFORCE_PHASE = 0
ATTACK_PHASE = 1
TRANSFER_PHASE = 2

def roll_dice():
    return random.randint(1, 6)

class Game:
    def __init__(self, num_players, debug=True, objectives_enabled=True):
        self.debug = debug
        self.objectives_enabled = objectives_enabled
        self.num_players = num_players
        self.players = self.setup_players()
        self.board = self.setup_board()
        self.cards = self.setup_territory_cards()
        self.current_card_troop_exchange = 4
        self.current_player_index = 0
        self.current_phase = REINFORCE_PHASE

    def setup_players(self):
        players = []
        colors = ['Azul', 'Amarelo', 'Vermelho', 'Cinza', 'Roxo', 'Verde']
        assert (self.num_players <= 6)
        objectives_list = objectives.copy()
        if self.debug:
            for i in range(self.num_players):
                player_name = input(f"\nEnter {i+1} Player's name: ")
                color = human_selector(colors, "\nCores disponíveis:","\nSelecione uma cor para o jogador:", False)
                colors.remove(color)
                objective = random.choice(objectives_list)
                objectives_list.remove(objective)
                input("\nPress enter to see your objective\n")
                debug_print(objectives_descriptions[objective])
                input("\nPress enter to hide your objective\n")
                os.system('cls' if os.name == 'nt' else 'clear')
                players.append(HumanPlayer(player_name, color, objective))
        else:
            for i in range(self.num_players):
                player_name = f"ai_{i}"
                color = random.choice(colors)
                objective = random.choice(objectives_list)
                objectives_list.remove(objective)
                players.append(AIPlayer(player_name, color, objective))
        return players
        


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
            debug_print(territory)

    def conquer_territory(self, attacker, defender):
        defender.owner.remove_territory(defender)
        attacker.owner.add_territory(defender)

        defender.owner = attacker.owner

        transfered_troops = input(f"\n{attacker.owner.name}, choose a number of troops to transfer (between 1 and {attacker.troops - 1}): ")
        transfered_troops = int(transfered_troops)
        transfered_troops = min(max(1, transfered_troops), 3)

        # Guarantee that at least 1 troop remains in each territory
        transfered_troops = max(1, attacker.troops - transfered_troops)
        debug_print(f"\nTransfered troops:{transfered_troops}")

        defender.troops = transfered_troops
        attacker.troops = attacker.troops - transfered_troops

    def attack(self, attacker, defender):
        # Simulate dice rolls and determine the outcome of the battle
        attacker_dice = [roll_dice() for _ in range(min(attacker.troops - 1, 3))]
        defender_dice = [roll_dice() for _ in range(min(defender.troops, 3))]

        attacker_dice.sort(reverse=True)
        defender_dice.sort(reverse=True)

        debug_print(f"{attacker.owner} rolled: {attacker_dice}")
        debug_print(f"{defender.owner} rolled: {defender_dice}")

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
            debug_print(f"{attacker.name} won the battle and conquered {defender.name}!")
            return True
        else:
            debug_print(f"{defender.name} successfully defended against {attacker.name}'s attack!")
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
        
        debug_print(f"{player.name} exchanged cards for {troops_to_reinforce} troops.")

        # Placing troops on owned territories
        debug_print(f"\n--- {player.name}'s Reinforcement Phase ---")
        player.place_troops(troops_to_reinforce)


    def start_round(self, player):
        player.round_base_placement(len(player.territories) // 2)

        conquered_continents = verify_conquered_continents(player)
        if conquered_continents:
            for continent in conquered_continents:
                debug_print("\nYou have a bonus for conquering an entire continent")
                troops_to_add = continent_to_troops(continent)
                player.place_troops(troops_to_add, continent)

        while True:
            if not player.is_human:
                self.reinforce(player)
                break
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
                debug_print("\nTry again")

    def attack_phase(self, player):
        # Assuming a manual attack; you can add more sophisticated game mechanics later
        attacker_territory = human_selector(player.territories,
                                            "\nYour territories:",
                                            f"\n{player.name}, choose a territory to attack from (0 to cancel): ",
                                            allow_zero=True)
        if attacker_territory == 0:
            return False
        attacker_territory = next((t for t in self.board if t == attacker_territory and t.owner == player), None)

        if not attacker_territory:
            debug_print("\nInvalid territory selection. Try again.")
            return False

        defender_territory = human_selector([t for t in attacker_territory.neighbors if t.owner != player],
                                            "\nLinked territories:",
                                            f"\n{player.name}, choose a territory to attack (0 to cancel): ",
                                            allow_zero=True)
        if defender_territory == 0:
            return False
        defender_territory = next((t for t in self.board if t == defender_territory and t in attacker_territory.neighbors), None)

        if not defender_territory:
            debug_print("Invalid target territory selection. Try again.")
            return False

        return self.attack(attacker_territory, defender_territory)

    def transfer_phase(self, player):
        # Assuming a manual attack; you can add more sophisticated game mechanics later
        giver_territory = human_selector(player.territories,
                                            "\nYour territories:",
                                            f"\n{player.name}, choose a territory to transfer from (0 to cancel): ",
                                            allow_zero=True)
        if giver_territory == 0:
            return False
        giver_territory = next((t for t in self.board if t == giver_territory and t.owner == player), None)

        receiver_territory = human_selector([t for t in giver_territory.neighbors if t.owner == player],
                                            "\nLinked territories:",
                                            f"\n{player.name}, choose a territory to transfer to (0 to cancel): ",
                                            allow_zero=True)
        if receiver_territory == 0:
            return False
        receiver_territory = next((t for t in self.board if t == receiver_territory and t in giver_territory.neighbors), None)

        if not receiver_territory:
            debug_print("Invalid target territory selection. Try again.")
            return False

        return self.transfer_troops(receiver_territory, receiver_territory)



    def play_round(self):
        current_player = self.players[self.current_player_index]

        if self.current_phase == REINFORCE_PHASE:
            self.start_round(current_player)

            debug_print(f"\n--- {current_player.name}'s Turn ---")
            self.display_board()

            self.current_attack_success = False
            self.current_phase = ATTACK_PHASE

            return
        
        if self.current_phase == ATTACK_PHASE:

            self.current_attack_success = self.attack_phase(current_player) or self.current_attack_success

            try:
                continue_attacking = int(input("\nDo you wish to continue attacking? 0=no, 1=yes\n"))
            except ValueError:
                debug_print('Invalid Selection, assuming False')
                continue_attacking = 0

            if not continue_attacking:
                # Give cards if one capture is done
                if self.current_attack_success:
                    debug_print("Card drawn")
                    current_player.cards.append(self.draw_card())
                
                self.current_phase = TRANSFER_PHASE
            
            return
        
        if self.current_phase == TRANSFER_PHASE:

            self.transfer_phase(current_player)

            try:
                continue_transfering = int(input("\nDo you wish to continue transfering? 0=no, 1=yes\n"))
            except ValueError:
                debug_print('Invalid Selection, assuming False')
                continue_transfering = 0

            if not continue_transfering:

                # Check if the game is over
                if self.objectives_enabled:
                    if check_win(current_player, self.players):
                        debug_print(f"\nCongratulations! {current_player.name} won the game!")
                        return
                else:
                    if simple_check_win(current_player, self.players):
                        return

                self.current_phase = REINFORCE_PHASE
                self.current_player_index = (self.current_player_index + 1) % len(self.players)

            return
            # return [self.current_player_index, state]

        


if __name__ == "__main__":
    num_players = int(input("Type in the number of players "))
    war_game = Game(debug=True, num_players=num_players)
    while True:
        war_game.play_round()
        #current_player, state = war_game.play_round()
        #war_game.play_action()

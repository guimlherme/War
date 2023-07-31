import os
import random
from typing import List
from game.players.player import Player
from game.players.human_player import HumanPlayer
from game.players.ai_player import AIPlayer

from game.territories import Board, Territory, TerritoryCard
from game.utils import human_selector, debug_print
from game.win_conditions import check_win, simple_check_win, objectives, objectives_descriptions
import numpy as np

REINFORCE_PHASE = 0
ATTACK_PHASE = 1
TRANSFER_PHASE = 2

MAX_ACTIONS_PER_ROUND = 30

def roll_dice():
    return random.randint(1, 6)

class Game:
    def __init__(self, num_players: int, debug:bool=True, objectives_enabled:bool=True):
        self.debug: bool = debug
        self.objectives_enabled: bool = objectives_enabled
        self.num_players: int = num_players
        self.players: List[Player] = self.setup_players()
        self.board: Board = self.setup_board()
        self.cards: List[TerritoryCard] = self.setup_territory_cards()
        self.current_card_troop_exchange: int = 4
        self.current_player_index: int = 0
        self.current_phase = REINFORCE_PHASE
        self.action_counter = 0

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

        board = Board()

        territories = board.territories_data.copy()

        random.shuffle(territories)
        split_territories = np.array_split(territories, len(self.players))
        for k, player in enumerate(self.players):
            for territory in split_territories[k].tolist():
                territory.owner = player
                player.add_territory(territory)
        
        for player in self.players:
            player.set_board(board)

        return board
    
    def setup_territory_cards(self):
        # Create TerritoryCard objects for each country
        territory_cards = [TerritoryCard(territory.name, territory.shape) 
                           for territory in self.board]
        return territory_cards

    def display_board(self):
        for territory in self.board:
            debug_print(territory)

    def get_state(self, player):
        state = []
        state.append(self.current_phase)
        for territory in self.board:
            state.append(player.players_dict(territory.owner))
            state.append(territory.troops)
        return state
    
    def get_last_reward(self, player_index):
        player = self.players[player_index]

        if player.is_human:
            print("Something went wrong. Tried to calculate reward to a human")
            return 0
        
        # Next lines are done in win_condition to facilitate checking
        # if player.has_died:
        #     return 0

        reward = -0.1 # Base reward is negative to prevent stalling

        if player.has_conquered: # Prize for gaining a card
            reward += 1
            player.has_conquered = False

        if player.has_won:
            reward += 500
        
        reward += player.reward_parcel
        player.reward_parcel = 0
        
        #TODO: Maybe use len(player.territories) instead of this
        reward += player.calculate_territory_change()

        return reward

    def set_ai_action(self, player_index, action):

        player = self.players[player_index]

        if player.is_human:
            print("Something went wrong. Tried to set action of a human")
            return 0
        
        player.action = action

    def conquer_territory(self, attacker, defender):
        defender.owner.remove_territory(defender)
        attacker.owner.add_territory(defender)

        if len(defender.owner.territories) == 0:
            defender.owner.has_died = True

        defender.owner = attacker.owner

        max_transferable_troops = min(attacker.troops - 1, 3)

        if attacker.owner.is_human:
            transfered_troops = input(f"\n{attacker.owner.name}, choose a number of troops to transfer (between 1 and {max_transferable_troops}): ")
            transfered_troops = int(transfered_troops)
            transfered_troops = min(max(1, transfered_troops), max_transferable_troops)
        else:
            # TODO: let AI decide this
            transfered_troops = max_transferable_troops

        debug_print(f"\nTransfered troops:{transfered_troops}")

        defender.troops = transfered_troops
        attacker.troops -= transfered_troops

    def attack(self, attacker: Territory, defender: Territory):
        """Performs the attack action

        Args:
            attacker (Territory): Territory that is attacking
            defender (Territory): Terrtory that is defending

        Returns:
            The tuple (attack_success, continue_attacking)
            attack_success indicates if the attack was successful, and
            continue_attacking indicates if another attack prompt will be issued
        """
        # Simulate dice rolls and determine the outcome of the battle
        attacker_dice = [roll_dice() for _ in range(min(attacker.troops - 1, 3))]
        defender_dice = [roll_dice() for _ in range(min(defender.troops, 3))]

        attacker_dice.sort(reverse=True)
        defender_dice.sort(reverse=True)

        debug_print(f"{attacker.owner.name} rolled: {attacker_dice}")
        debug_print(f"{defender.owner.name} rolled: {defender_dice}")

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
            return True, True
        else:
            debug_print(f"{defender.name} successfully defended against {attacker.name}'s attack!")
            return False, True

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
            return 0
        
        debug_print(f"{player.name} exchanged cards for {troops_to_reinforce} troops.")

        return troops_to_reinforce

    
    def start_round(self, player):
        debug_print(f"\n {player.name}'s turn:\n")
        conquered_continents = self.board.verify_conquered_continents(player)
        if conquered_continents:
            for continent in conquered_continents:
                debug_print("\nYou have a bonus for conquering an entire continent")
                troops_to_add = self.board.continent_to_troops(continent)
                # TODO: Make AI control this, too
                player.place_troops(troops_to_add, continent)
        
        round_base_placement = len(player.territories) // 2

        reinforce_num_troops = 0
        while True:
            if not player.is_human:
                reinforce_num_troops = self.reinforce(player)
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
        
        player.remaining_troops_to_place = round_base_placement + reinforce_num_troops


    def reinforcement_phase(self, player):
        player.place_troops(player.remaining_troops_to_place)
        return player.remaining_troops_to_place

    def attack_phase(self, player: Player):
        """Starts the attack phase

        Args:
            player (Player): player that will be attacking

        Returns:
            The tuple (attack_success, continue_attacking)
            attack_success indicates if the attack was successful, and
            continue_attacking indicates if another attack prompt will be issued
        """
        
        attack_intention = player.prepare_attack(self.board)
        
        if isinstance(attack_intention[0], bool):
            return attack_intention
        
        attacker_territory, defender_territory = attack_intention

        # Sanity check
        if attacker_territory.owner != player or defender_territory.owner == player:
            return False, True

        return self.attack(attacker_territory, defender_territory)

    def transfer_phase(self, player):
        
        transfer_intention = player.prepare_transfer(self.board)

        if isinstance(transfer_intention, bool):
            # Continue transfering or not
            return transfer_intention
        
        giver_territory, receiver_territory = transfer_intention

        # Sanity check
        if giver_territory.owner != player or receiver_territory.owner != player:
            return True

        return self.transfer_troops(giver_territory, receiver_territory)

    def transfer_troops(self, giver_territory, receiver_territory):

        # TODO: implement a rule such that the same troop can't be transfered twice in the same turn
        max_transferable_troops = giver_territory.troops - 1

        if giver_territory.owner.is_human:
            transfered_troops = input(f"\n{giver_territory.owner.name}, choose a number of troops to transfer (between 1 and {max_transferable_troops}): ")
            transfered_troops = int(transfered_troops)
            transfered_troops = min(max(1, transfered_troops), max_transferable_troops)
        else:
            transfered_troops = 1

        debug_print(f"\nTransfered troops:{transfered_troops}")

        receiver_territory.troops += transfered_troops
        giver_territory.troops -= transfered_troops
        return True

    def change_player(self):
        self.current_player_index = (self.current_player_index + 1) % len(self.players)
        current_player = self.players[self.current_player_index]
        # Make sure that next player is alive
        while current_player.has_died:
            self.current_player_index = (self.current_player_index + 1) % len(self.players)
            current_player = self.players[self.current_player_index]


    def play_round(self):
        current_player = self.players[self.current_player_index]
        if self.action_counter >= MAX_ACTIONS_PER_ROUND:
            self.change_player()
            return [self.current_player_index, self.get_state(current_player)]
        self.action_counter += 1

        if self.current_phase == REINFORCE_PHASE:
            if current_player.remaining_troops_to_place == 0:
                self.start_round(current_player)
            remaining_troops_to_place = self.reinforcement_phase(current_player)

            if remaining_troops_to_place == 0:
                debug_print(f"\n--- {current_player.name}'s Turn ---")
                if self.debug:
                    self.display_board()

                self.current_phase = ATTACK_PHASE
                self.current_attack_success = False

            return [self.current_player_index, self.get_state(current_player)]
        
        if self.current_phase == ATTACK_PHASE:

            attack_success, continue_attacking = self.attack_phase(current_player)

            if attack_success and not self.current_attack_success:
                # This is to make sure that the reward is immediate
                current_player.has_conquered = True
                self.current_attack_success = True

            if not continue_attacking:
                # Give cards if one capture is done
                if self.current_attack_success:
                    debug_print("Card drawn")
                    current_player.cards.append(self.draw_card())
                
                self.current_phase = TRANSFER_PHASE
            
            return [self.current_player_index, self.get_state(current_player)]
        
        if self.current_phase == TRANSFER_PHASE:

            continue_transfering = self.transfer_phase(current_player)

            if not continue_transfering:

                # Check if the game is over
                if self.objectives_enabled:
                    if check_win(current_player, self.players, self.board):
                        debug_print(f"\nCongratulations! {current_player.name} won the game!")
                        current_player.has_won = True
                        return [None, self.get_state(current_player)]
                else:
                    if simple_check_win(current_player, self.players, self.board):
                        current_player.has_won = True
                        return [None, self.get_state(current_player)]

                # Do the reward counting updates

                self.action_counter = 0
                self.current_phase = REINFORCE_PHASE

                self.change_player()

            return [self.current_player_index, self.get_state(current_player)]

        


if __name__ == "__main__":
    num_players = int(input("Type in the number of players "))
    war_game = Game(num_players=num_players, debug=True)
    while True:
        war_game.play_round()
        #next_player, state = war_game.play_round()
        #war_game.set_ai_action(next_player, action)

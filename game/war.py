import os
import random
from math import ceil
from typing import List
from game.players.player import Player
from game.players.human_player import HumanPlayer
from game.players.ai_player import AIPlayer, AI_TROOP_SELECTION_FACTOR

from game.territories import Board, Territory, TerritoryCard
from game.utils import human_selector, debug_print, bool_human_selector
from game.win_conditions import check_win, simple_check_win, objectives
import numpy as np

REINFORCEMENT_PHASE = 0
ATTACK_PHASE = 1
TRANSFER_PHASE = 2

MAX_ACTIONS_PER_ROUND = 40
MAX_ACTIONS_PER_MATCH = 4000

BASE_REWARD = -0.01 # Base reward is slightly negative to prevent stalling
CARD_REWARD = 0.5
TERRITORIAL_CHANGE_FACTOR = 1.0
VICTORY_REWARD = 50.0
SPEED_FACTOR = 1.0
MAX_ROUND_FOR_BONUS = 1.0 * (MAX_ACTIONS_PER_MATCH / MAX_ACTIONS_PER_ROUND) # Max actions per round is rarely used completely

all_colors = ['Azul', 'Amarelo', 'Vermelho', 'Cinza', 'Roxo', 'Verde']

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
        self.current_phase = REINFORCEMENT_PHASE
        self.round_actions = set()
        self.round_action_counter = 0
        self.match_action_counter = 0
        self.current_round_number = 0 # This counts each individual turn
        self.ended_in_objective = False
        # If you ever modify this, remember to reset it

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
                debug_print(objective.description)
                input("\nPress enter to hide your objective\n")
                os.system('cls' if os.name == 'nt' else 'clear')
                players.append(HumanPlayer(player_name, color, objective))
        else:
            for i in range(self.num_players):
                player_name = f"ai_{i}"
                color = random.choice(colors)
                colors.remove(color)
                objective = random.choice(objectives_list)
                objectives_list.remove(objective)
                players.append(AIPlayer(player_name, color, objective))
        return players
    
    @property
    def current_player(self):
        return self.players[self.current_player_index]

    def setup_board(self, board = Board()):

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
        territory_cards = [TerritoryCard(territory, territory.shape) 
                           for territory in self.board]
        return territory_cards
    
    def reset(self):
        del self.players
        self.players = self.setup_players()

        for territory in self.board:
            territory.troops = 1
            territory.owner = None
        self.board = self.setup_board(self.board)

        del self.cards
        self.cards = self.setup_territory_cards()

        self.current_card_troop_exchange = 4
        self.current_player_index = 0
        self.current_phase = REINFORCEMENT_PHASE
        self.round_actions = set()
        self.round_action_counter = 0
        self.match_action_counter = 0
        self.current_round_number = 0
        self.ended_in_objective = False

        return self.get_state(self.current_player)

    def setup_from_state(self, state): #WIP
        
        main_player = self.players[0] #FIXME

        self.current_phase = state[0]
        obj_id = state[1]
        objective = next((o for o in objectives if o.id == obj_id), None)
        assert objective != None

        colors_players = {main_player.players_dict(p): p for p in self.players}

        territories_owners = state[2::2]
        territories_troops = state[3::2]
        for i, territory in enumerate(self.board):
            territory.owner = colors_players[territories_owners[i]]
            if territory.owner == 0:
                territory.troops = territories_troops[i]
            else:
                territory.troops = -territories_troops[i]



    def display_board(self):
        for territory in self.board:
            debug_print(territory)

    def get_state(self, player: Player):
        state = []
        state.append(self.current_phase)
        state.append(player.objective.id)
        for territory in self.board:
            state.append(player.players_dict(territory.owner))
            if territory.owner == 0:
                state.append(territory.troops)
            else: # Help the algorithm to differentiate other players' territories
                state.append(-territory.troops)
        return state
    
    def get_last_reward(self, player):

        if player.is_human:
            print("Something went wrong. Tried to calculate reward to a human")
            return 0
        
        if player.has_died:
            raise ValueError('Tried to get reward for a dead player')

        reward = BASE_REWARD

        if player.has_conquered: # Prize for gaining a card
            reward += CARD_REWARD
            player.has_conquered = False

        if player.has_won:
            reward += VICTORY_REWARD + max(0, SPEED_FACTOR * (MAX_ROUND_FOR_BONUS - self.current_round_number))
        
        reward += player.reward_parcel
        player.reward_parcel = 0
        
        reward += TERRITORIAL_CHANGE_FACTOR * player.calculate_territory_change()

        return reward

    def get_match_action_counter(self):
        return self.match_action_counter

    # def get_valid_actions_from_state(self, state):
    #     #FIXME
    #     current_phase = state[0]
    #     current_player = 0
    #     territories = state[1:]
    #     owners = []
    #     num_troops = []
    #     for territory_i in range(0, len(territories), 2):
    #         owners.append(territories[territory_i])
    #         num_troops.append(territories[territory_i + 1])

    #     if current_phase == 0:
    #         actions_table = [False] # Passing isn't an option
    #     else:
    #         actions_table = [True] # Passing is an option
        
    #     for i, territory in enumerate(self.board):
    #         if current_phase == REINFORCEMENT_PHASE:
    #             if owners[i] == current_player: # If current player is the owner
    #                 actions_table.append(True)
    #             else:
    #                 actions_table.append(False)
    #         elif current_phase == ATTACK_PHASE:
    #             if owners[i] == current_player:
    #                 actions_table.append(False)
    #             else:
    #                 if any([(owners[n.index] == current_player and num_troops[n.index] >= 2) for n in territory.neighbors]):
    #                     actions_table.append(True)
    #                 else:
    #                     actions_table.append(False)
    #         elif current_phase == TRANSFER_PHASE:
    #             if owners[i] != current_player:
    #                 actions_table.append(False)
    #             else:
    #                 if any([(owners[n.index] == current_player and num_troops[n.index] >= 2) for n in territory.neighbors]):
    #                     actions_table.append(True)
    #                 else:
    #                     actions_table.append(False)
    #     return actions_table


    def get_valid_actions_table(self):
        # Remember to change the function above if you ever change this function
        if self.current_phase == 0:
            actions_table = [False] # Passing isn't an option
        else:
            actions_table = [True] # Passing is an option
        
        for territory in self.board:
            if self.current_phase == REINFORCEMENT_PHASE:
                if territory.owner == self.current_player: # If current player is the owner
                    actions_table.append(True)
                else:
                    actions_table.append(False)
            elif self.current_phase == ATTACK_PHASE:
                if territory.owner == self.current_player:
                    actions_table.append(False)
                else:
                    if any([(n.owner == self.current_player and n.troops >= 2) for n in territory.neighbors]):
                        actions_table.append(True)
                    else:
                        actions_table.append(False)
            elif self.current_phase == TRANSFER_PHASE:
                if territory.owner != self.current_player:
                    actions_table.append(False)
                else:
                    if any([(n.owner == self.current_player and n.troops >= 2) for n in territory.neighbors]):
                        actions_table.append(True)
                    else:
                        actions_table.append(False)
        return actions_table

                

    def set_ai_action(self, player_index, action):

        player = self.players[player_index]

        if player.is_human:
            print("Something went wrong. Tried to set action of a human")
            return 0
        
        player.action = action

    def conquer_territory(self, attacker: Territory, defender: Territory):
        defender.owner.remove_territory(defender)
        attacker.owner.add_territory(defender)

        if len(defender.owner.territories) == 0:
            defender.owner.has_died = True
            attacker.owner.killed_list.append(defender.owner)
            print('Someone has died! ', attacker.owner.name, ' killed ', defender.owner.name)

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
        # Check if the player is in godmode and controls the results of the attacks
        if attacker.owner.godmode:
            return attacker.owner.godmode_attack(attacker, defender)
        
        initial_attacker_troops = attacker.troops
        while True:
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
                return True, True #TODO: refactor this logic
            elif attacker.troops <= 1:
                if attacker.troops <= 0: raise ValueError("Attacker ended up with 0 troops")
                debug_print(f"{defender.name} successfully defended against {attacker.name}'s attack!")
                return False, True
            else:
                if attacker.owner.is_human:
                    continue_attacking = bool_human_selector("Do you wish to continue attacking? 0=no, 1=yes")
                else:
                    continue_attacking = (attacker.troops >= (1-AI_TROOP_SELECTION_FACTOR) * initial_attacker_troops )

                if continue_attacking:
                    continue
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

        # TODO: make exchange controllable (e.g. choose the cards which will be traded)
        exchange_successful = False
        unique_cards = 0
        card_types = ["Circle", "Square", "Triangle"]
        cards_dict = player.count_card_types()

        for card_type in card_types:
            if cards_dict[card_type] >= 1:
                unique_cards += 1
            if cards_dict[card_type] >= 3:
                for _ in range(3):
                    removed_card = player.remove_one_card_from_type(card_type)
                    if removed_card.territory.owner == player:
                        debug_print(f"\nYou exchanged a card of {removed_card.territory.name}! 2 extra troops were added.")
                        removed_card.territory.troops += 2
                    self.cards.append(removed_card)
                    random.shuffle(self.cards)
                    exchange_successful = True
        
        # Prevent double exchanges
        if unique_cards == 3 and not exchange_successful:
            for card_type in card_types:
                removed_card = player.remove_one_card_from_type(card_type)
                if removed_card.territory.owner == player:
                    debug_print(f"\nYou exchanged a card of {removed_card.territory.name}! 2 extra troops were added.")
                    removed_card.territory.troops += 2
                self.cards.append(removed_card)
                random.shuffle(self.cards)
            exchange_successful = True

        if exchange_successful:
            num_troops = self.current_card_troop_exchange
            self.current_card_troop_exchange += 2
            return num_troops     

        return 0

    def reinforce_by_cards(self, player):
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
        if not player.is_human:
            reinforce_num_troops = self.reinforce_by_cards(player)
        else:
            bool_reinforce = bool_human_selector("Do you want to exchange your cards, if possible? 0=no, 1=yes ")
            if bool_reinforce:
                self.reinforce_by_cards(player)
        
        player.remaining_troops_to_place = round_base_placement + reinforce_num_troops


    def reinforcement_phase(self, player):
        player.place_troops(player.remaining_troops_to_place)
        return player.remaining_troops_to_place

    def attack_phase(self, player: Player):
        """Starts the attack phase

        Args:
            player (Player): player that will be attacking

        Returns:
            #TODO: This is simply outrageous, but works
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

    def transfer_troops(self, giver_territory: Territory, receiver_territory: Territory):

        # TODO: implement a rule such that the same troop can't be transfered twice in the same turn
        max_transferable_troops = giver_territory.troops - 1

        if giver_territory.owner.is_human:
            transfered_troops = input(f"\n{giver_territory.owner.name}, choose a number of troops to transfer (between 1 and {max_transferable_troops}): ")
            transfered_troops = int(transfered_troops)
        else:
            transfered_troops = ceil(AI_TROOP_SELECTION_FACTOR * giver_territory.troops)

        transfered_troops = min(max(1, transfered_troops), max_transferable_troops)

        debug_print(f"\nTransfered troops:{transfered_troops}")

        receiver_territory.troops += transfered_troops
        giver_territory.troops -= transfered_troops
        return True

    def change_player(self):
        self.round_action_counter = 0
        self.current_round_number += 1 # This counts each individual turn
        self.current_player_index = (self.current_player_index + 1) % len(self.players)
        # Make sure that next player is alive
        while self.current_player.has_died:
            self.current_player_index = (self.current_player_index + 1) % len(self.players)


    def play_round(self, action=None):
        if action != None:
            if self.current_player.is_human:
                raise ValueError("Action issued to human player")
            self.current_player.action = action
            target_territory = self.current_player.get_action()
            if target_territory is not None:
                self.round_actions.add((self.current_phase, target_territory.name, target_territory.troops))

        if action == None:
            if not self.current_player.is_human:
                raise ValueError("No action issued to AI player")

        if self.round_action_counter >= MAX_ACTIONS_PER_ROUND:
            self.change_player()
            return [self.current_player_index, self.get_state(self.current_player), self.get_last_reward(self.current_player)]
        self.round_action_counter += 1

        if self.match_action_counter >= MAX_ACTIONS_PER_MATCH:
            return [None, self.get_state(self.current_player), 0]
        self.match_action_counter += 1

        if self.current_phase == REINFORCEMENT_PHASE:
            if self.current_player.remaining_troops_to_place == 0:
                self.start_round(self.current_player)
            remaining_troops_to_place = self.reinforcement_phase(self.current_player)

            if remaining_troops_to_place < 0:
                raise ValueError('Negative remaining troops to place')

            if remaining_troops_to_place == 0:
                debug_print(f"\n--- {self.current_player.name}'s Turn ---")
                if self.debug:
                    self.display_board()

                self.current_phase = ATTACK_PHASE
                self.current_attack_success = False

            return [self.current_player_index, self.get_state(self.current_player), self.get_last_reward(self.current_player)]
        
        if self.current_phase == ATTACK_PHASE:

            attack_success, continue_attacking = self.attack_phase(self.current_player)

            if attack_success and not self.current_attack_success:
                # This is to make sure that the reward is immediate
                self.current_player.has_conquered = True
                self.current_attack_success = True

            if not continue_attacking:
                # Give cards if one capture is done
                if self.current_attack_success:
                    debug_print("Card drawn")
                    self.current_player.cards.append(self.draw_card())
                
                self.current_phase = TRANSFER_PHASE
            
            return [self.current_player_index, self.get_state(self.current_player), self.get_last_reward(self.current_player)]
        
        if self.current_phase == TRANSFER_PHASE:

            continue_transfering = self.transfer_phase(self.current_player)

            if not continue_transfering:

                # Check if the game is over
                if self.objectives_enabled:
                    if check_win(self.current_player, self.players, self.board):
                        debug_print(f"\nCongratulations! {self.current_player.name} won the game!")
                        self.current_player.has_won = True
                        self.ended_in_objective = True
                        for p in self.players:
                            if p != self.current_player:
                                p.has_died = True
                        return [None, self.get_state(self.current_player), self.get_last_reward(self.current_player)]
                else:
                    if simple_check_win(self.current_player, self.players, self.board):
                        self.current_player.has_won = True
                        self.ended_in_objective = True
                        for p in self.players:
                            if p != self.current_player:
                                p.has_died = True
                        return [None, self.get_state(self.current_player), self.get_last_reward(self.current_player)]

                # Do the reward counting updates

                self.round_action_counter = 0
                self.current_phase = REINFORCEMENT_PHASE

                # print(self.current_player.name, len(self.current_player.territories))
                # print(self.round_actions)
                # input()

                self.round_actions.clear()
                self.change_player()

            return [self.current_player_index, self.get_state(self.current_player), self.get_last_reward(self.current_player)]

        


if __name__ == "__main__":
    num_players = int(input("Type in the number of players "))
    war_game = Game(num_players=num_players, debug=True)
    while True:
        war_game.play_round()
        #next_player, state = war_game.play_round()
        #war_game.set_ai_action(next_player, action)

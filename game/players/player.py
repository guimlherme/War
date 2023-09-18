from game.territories import Territory

class Player:
    def __init__(self, name, color, objective, is_human, godmode):
        self.name = name
        self.color = color
        self.objective = objective
        self.board = None
        self.territories = []
        self.cards = []
        self.killed_list = []
        self.last_territory_len = 0
        self.reward_parcel = 0
        self.is_human = is_human
        self.godmode = godmode
        self.has_died = False
        self.has_conquered = False
        self.has_won = False
        self.remaining_troops_to_place = 0

    def set_board(self, board):
        self.board = board

    def players_dict(self, player):
        # Attributes a number to each player based on this player's referential
        # For now, it's only 0=me, 1=others
        return 0 if player==self else 1
    
    def calculate_territory_change(self):
        if self.last_territory_len == 0:
            self.last_territory_len = len(self.territories)
            return 0
        change = len(self.territories) - self.last_territory_len
        self.last_territory_len = len(self.territories)
        return change

    def get_card_types(self):
        return [card.get_card_type() for card in self.cards]
    
    def count_card_types(self):
        counting = {
            "Circle": 0,
            "Square": 0,
            "Triangle": 0
        }
        for card_type in self.get_card_types():
            counting[card_type] += 1
        return counting

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
    
    def prepare_attack(*args):
        raise NotImplementedError
    
    def prepare_transfer(*args):
        raise NotImplementedError
    
    def godmode_attack(self, attacker: Territory, defender: Territory):
        assert self == attacker
        while True:
            try:
                print("Did the attack succeed? 0=False, 1=True")
                succeed = bool(int(input()))
                print("Type in the final number of troops in the attacker territory:")
                attacker_final_troops = int(input())
                print("Type in the final number of troops in the defender territory:")
                defender_final_troops = int(input())
                break
            except:
                print("Try Again")
        
        attacker.troops = attacker_final_troops
        defender.troops = defender_final_troops

        if succeed:
            defender.owner.remove_territory(defender)
            attacker.owner.add_territory(defender)

            if len(defender.owner.territories) == 0:
                defender.owner.has_died = True
                attacker.owner.killed_list.append(defender.owner)
            defender.owner = attacker.owner

            return True, True
        
        else:
            return False, True

        
    

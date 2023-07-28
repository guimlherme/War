class Player:
    def __init__(self, name):
        self.name = name
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
            print("Your territories:")
            for i, territory in enumerate(self.territories):
                print(f"{i+1}. {territory}")

            try:
                choice = int(input("Enter the number of the territory to put troops on (0 to finish): "))
                if choice < 0 or choice > len(self.territories):
                    raise ValueError

                if choice == 0:
                    break

                selected_territory = self.territories[choice - 1]
                troops_to_place = int(input(f"How many troops to place on {selected_territory.name}? "))

                if troops_to_place <= remaining_troops:
                    selected_territory.troops += troops_to_place
                    remaining_troops -= troops_to_place
                else:
                    print("You don't have enough troops remaining.")
            except (ValueError, IndexError):
                print("Invalid input. Please try again.")

    def round_base_placement(self, num_troops):
        self.place_troops(num_troops)

#TODO: Use the real values
territories_data = [
            ("Alaska", "Circle"), ("Northwest Territory", "Square"), ("Greenland", "Triangle"),
            ("Alberta", "Square"), ("Ontario", "Circle"), ("Quebec", "Triangle"),
            ("Western United States", "Circle"), ("Eastern United States", "Triangle"), ("Central America", "Square"),
            ("Venezuela", "Square"), ("Brazil", "Triangle"), ("Peru", "Circle"), ("Argentina", "Circle"),
            ("Iceland", "Circle"), ("Scandinavia", "Triangle"), ("Great Britain", "Square"),
            ("Northern Europe", "Square"), ("Western Europe", "Circle"), ("Southern Europe", "Circle"), ("Ukraine", "Triangle"),
            ("North Africa", "Triangle"), ("Egypt", "Circle"), ("East Africa", "Square"), ("Congo", "Circle"),
            ("South Africa", "Square"), ("Madagascar", "Triangle"),
            ("Ural", "Square"), ("Siberia", "Triangle"), ("Yakutsk", "Square"), ("Kamchatka", "Triangle"),
            ("Irkutsk", "Circle"), ("Mongolia", "Circle"), ("Japan", "Square"), ("Afghanistan", "Triangle"),
            ("Middle East", "Square"), ("India", "Triangle"), ("China", "Circle"), ("Siam", "Square"),
            ("Indonesia", "Circle"), ("New Guinea", "Triangle"),
            ("Western Australia", "Circle"), ("Eastern Australia", "Square"),
]

def add_links(board):
    # Define the links between territories here
    # For example:
    board[0].add_neighbor(board[1])  # Alaska is linked to Northwest Territory
    board[1].add_neighbor(board[0])  # Northwest Territory is linked to Alaska
    # ...
    #TODO: complete this


class Territory:
    def __init__(self, name, owner=None, troops=0):
        self.name = name
        self.owner = owner
        self.troops = troops
        self.neighbors = []

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)

    def __str__(self):
        links = [n.name for n in self.neighbors]
        return f"{self.name} (Owner: {self.owner.name}, Troops: {self.troops}, Links: {links})"
    
class TerritoryCard:
    def __init__(self, territory, card_type):
        self.territory = territory
        self.card_type = card_type

    def get_card_type(self):
        return self.card_type
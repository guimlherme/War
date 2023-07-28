#TODO: Use the real card shapes
territories_data = [
            # North America
            ("Alasca", "Circle"), ("Mackenzie", "Square"), ("Groelandia", "Triangle"),
            ("Vancouver", "Square"), ("Ottawa", "Circle"), ("Labrador", "Triangle"),
            ("California", "Circle"), ("Nova_York", "Triangle"), ("Mexico", "Square"),

            # South America
            ("Venezuela", "Square"), ("Brasil", "Triangle"), ("Peru", "Circle"), ("Argentina", "Circle"),

            # Europe
            ("Islandia", "Circle"), ("Inglaterra", "Triangle"), ("Suecia", "Square"),
            ("Moscou", "Square"), ("Franca", "Circle"), ("Alemanha", "Circle"), ("Polonia", "Triangle"),

            # Africa
            ("Argelia", "Triangle"), ("Egito", "Circle"), ("Sudao", "Square"), ("Congo", "Circle"),
            ("Africa_do_Sul", "Square"), ("Madagascar", "Triangle"),

            # Asia
            ("Omsk", "Square"), ("Dudinka", "Triangle"), ("Siberia", "Square"), ("Vladvostok", "Triangle"),
            ("Aral", "Circle"), ("Mongolia", "Circle"), ("Tchita", "Square"), ("Oriente_Medio", "Triangle"),
            ("India", "Square"), ("Vietna", "Triangle"), ("China", "Circle"), ("Japao", "Square"),


            ("Sumatra", "Circle"), ("Borneu", "Triangle"),
            ("Nova_Guine", "Circle"), ("Australia", "Square")
]

def add_edge(board, country1, country2):
    countries = [t.name for t in board]
    index1 = countries.index(country1)
    index2 = countries.index(country2)
    board[index1].add_neighbor(board[index2])


def add_links(board):
    add_edge(board, "Brasil", "Argentina")
    add_edge(board, "Brasil", "Venezuela")
    add_edge(board, "Brasil", "Peru")
    add_edge(board, "Argentina", "Peru")
    add_edge(board, "Venezuela", "Peru")

    # América do Norte
    add_edge(board, "Alasca", "Mackenzie")
    add_edge(board, "Alasca", "Vancouver")
    add_edge(board, "Mackenzie", "Groelandia")
    add_edge(board, "Mackenzie", "Ottawa")
    add_edge(board, "Mackenzie", "Vancouver")
    add_edge(board, "Groelandia", "Labrador")
    add_edge(board, "Vancouver", "Ottawa")
    add_edge(board, "Vancouver", "California")
    add_edge(board, "Ottawa", "Labrador")
    add_edge(board, "Nova_York", "Labrador")
    add_edge(board, "Ottawa", "California")
    add_edge(board, "Ottawa", "Nova_York")
    add_edge(board, "California", "Nova_York")
    add_edge(board, "California", "Mexico")
    add_edge(board, "Nova_York", "Mexico")

    # Europa
    add_edge(board, "Islandia", "Inglaterra")
    add_edge(board, "Inglaterra", "Suecia")
    add_edge(board, "Inglaterra", "Alemanha")
    add_edge(board, "Inglaterra", "Franca")
    add_edge(board, "Polonia", "Alemanha")
    add_edge(board, "Moscou", "Suecia")
    add_edge(board, "Moscou", "Polonia")
    add_edge(board, "Franca", "Alemanha")
    add_edge(board, "Franca", "Polonia")

    # Ásia
    add_edge(board, "Omsk", "Dudinka")
    add_edge(board, "Omsk", "Aral")
    add_edge(board, "Omsk", "Mongolia")
    add_edge(board, "Dudinka", "Siberia")
    add_edge(board, "Dudinka", "Tchita")
    add_edge(board, "Dudinka", "Mongolia")
    add_edge(board, "Siberia", "Tchita")
    add_edge(board, "Siberia", "Vladvostok")
    add_edge(board, "Tchita", "Vladvostok")
    add_edge(board, "Tchita", "Mongolia")
    add_edge(board, "Tchita", "China")
    add_edge(board, "Vladvostok", "China")
    add_edge(board, "Vladvostok", "Japao")
    add_edge(board, "Aral", "Oriente_Medio")
    add_edge(board, "Aral", "India")
    add_edge(board, "Aral", "China")
    add_edge(board, "Mongolia", "China")
    add_edge(board, "China", "Japao")
    add_edge(board, "China", "Vietna")
    add_edge(board, "China", "India")
    add_edge(board, "Vietna", "India")
    add_edge(board, "India", "Oriente_Medio")

    # África
    add_edge(board, "Argelia", "Egito")
    add_edge(board, "Argelia", "Sudao")
    add_edge(board, "Argelia", "Congo")
    add_edge(board, "Egito", "Sudao")
    add_edge(board, "Sudao", "Congo")
    add_edge(board, "Sudao", "Madagascar")
    add_edge(board, "Sudao", "Africa_do_Sul")
    add_edge(board, "Congo", "Africa_do_Sul")
    add_edge(board, "Africa_do_Sul", "Madagascar")

    # Oceania
    add_edge(board, "Sumatra", "Australia")
    add_edge(board, "Borneu", "Nova_Guine")
    add_edge(board, "Borneu", "Australia")
    add_edge(board, "Nova_Guine", "Australia")

    # Intercontinentais
    # América do Sul - América do Norte
    add_edge(board, "Venezuela", "Mexico")

    # América do Sul - África
    add_edge(board, "Brasil", "Argelia")

    # América do Norte - Europa
    add_edge(board, "Groelandia", "Islandia")

    # América do Norte - Asia
    add_edge(board, "Alasca", "Vladvostok")

    # Europa - Asia
    add_edge(board, "Polonia", "Oriente_Medio")
    add_edge(board, "Moscou", "Omsk")
    add_edge(board, "Moscou", "Aral")
    add_edge(board, "Moscou", "Oriente_Medio")

    # Europa - África
    add_edge(board, "Franca", "Argelia")
    add_edge(board, "Franca", "Egito")
    add_edge(board, "Polonia", "Egito")

    # Asia - África
    add_edge(board, "Oriente_Medio", "Egito")

    # Asia - Oceania
    add_edge(board, "India", "Sumatra")
    add_edge(board, "Vietna", "Borneu")

    return map


class Territory:
    def __init__(self, name, owner=None, troops=0):
        self.name = name
        self.owner = owner
        self.troops = troops
        self.neighbors = []

    def add_neighbor(self, neighbor):
        if neighbor not in self.neighbors:
            self.neighbors.append(neighbor)
            neighbor.neighbors.append(self)

    def __str__(self):
        links = [n.name for n in self.neighbors]
        return f"{self.name} (Owner: {self.owner.name}, Troops: {self.troops}, Links: {links})"
    
class TerritoryCard:
    def __init__(self, territory, card_type):
        self.territory = territory
        self.card_type = card_type

    def get_card_type(self):
        return self.card_type
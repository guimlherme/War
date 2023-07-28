
class Territory:
    def __init__(self, name, shape, owner=None, troops=1):
        self.name = name
        self.shape = shape
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

# North America
Alasca = Territory("Alasca", "Circle")
Mackenzie = Territory("Mackenzie", "Square")
Groelandia = Territory("Groelandia", "Triangle")
Vancouver = Territory("Vancouver", "Square")
Ottawa = Territory("Ottawa", "Circle")
Labrador = Territory("Labrador", "Triangle")
California = Territory("California", "Circle")
Nova_York = Territory("Nova_York", "Triangle")
Mexico = Territory("Mexico", "Square")

# South America
Venezuela = Territory("Venezuela", "Square")
Brasil = Territory("Brasil", "Triangle")
Peru = Territory("Peru", "Circle")
Argentina = Territory("Argentina", "Circle")

# Europe
Islandia = Territory("Islandia", "Circle")
Inglaterra = Territory("Inglaterra", "Triangle")
Suecia = Territory("Suecia", "Square")
Moscou = Territory("Moscou", "Square")
Franca = Territory("Franca", "Circle")
Alemanha = Territory("Alemanha", "Circle")
Polonia = Territory("Polonia", "Triangle")

# Africa
Argelia = Territory("Argelia", "Triangle")
Egito = Territory("Egito", "Circle")
Sudao = Territory("Sudao", "Square")
Congo = Territory("Congo", "Circle")
Africa_do_Sul = Territory("Africa_do_Sul", "Square")
Madagascar = Territory("Madagascar", "Triangle")

# Asia
Omsk = Territory("Omsk", "Square")
Dudinka = Territory("Dudinka", "Triangle")
Siberia = Territory("Siberia", "Square")
Vladvostok = Territory("Vladvostok", "Triangle")
Aral = Territory("Aral", "Circle")
Mongolia = Territory("Mongolia", "Circle")
Tchita = Territory("Tchita", "Square")
Oriente_Medio = Territory("Oriente_Medio", "Triangle")
India = Territory("India", "Square")
Vietna = Territory("Vietna", "Triangle")
China = Territory("China", "Circle")
Japao = Territory("Japao", "Square")

# Oceania
Sumatra = Territory("Sumatra", "Circle")
Borneu = Territory("Borneu", "Triangle")
Nova_Guine = Territory("Nova_Guine", "Circle")
Australia = Territory("Australia", "Square")

north_america= [Alasca, Mackenzie, Groelandia, Vancouver, Ottawa, Labrador, California, Nova_York, Mexico]
south_america = [Venezuela, Brasil, Peru, Argentina]
europe = [Islandia, Inglaterra, Suecia, Moscou, Franca, Alemanha, Polonia]
africa = [Argelia, Egito, Sudao, Congo, Africa_do_Sul, Madagascar]
asia = [Omsk, Dudinka, Siberia, Vladvostok, Aral, Mongolia, Tchita, Oriente_Medio, India, Vietna, China, Japao]
oceania = [Sumatra, Borneu, Nova_Guine, Australia]

continents = [north_america, south_america, europe, africa, asia, oceania]

territories_data = north_america + south_america + europe + africa + asia + oceania


def add_edge(country1, country2):
    country1.add_neighbor(country2)

add_edge(Brasil, Argentina)
add_edge(Brasil, Venezuela)
add_edge(Brasil, Peru)
add_edge(Argentina, Peru)
add_edge(Venezuela, Peru)

# América do Norte
add_edge(Alasca, Mackenzie)
add_edge(Alasca, Vancouver)
add_edge(Mackenzie, Groelandia)
add_edge(Mackenzie, Ottawa)
add_edge(Mackenzie, Vancouver)
add_edge(Groelandia, Labrador)
add_edge(Vancouver, Ottawa)
add_edge(Vancouver, California)
add_edge(Ottawa, Labrador)
add_edge(Nova_York, Labrador)
add_edge(Ottawa, California)
add_edge(Ottawa, Nova_York)
add_edge(California, Nova_York)
add_edge(California, Mexico)
add_edge(Nova_York, Mexico)

# Europa
add_edge(Islandia, Inglaterra)
add_edge(Inglaterra, Suecia)
add_edge(Inglaterra, Alemanha)
add_edge(Inglaterra, Franca)
add_edge(Polonia, Alemanha)
add_edge(Moscou, Suecia)
add_edge(Moscou, Polonia)
add_edge(Franca, Alemanha)
add_edge(Franca, Polonia)

# Ásia
add_edge(Omsk, Dudinka)
add_edge(Omsk, Aral)
add_edge(Omsk, Mongolia)
add_edge(Dudinka, Siberia)
add_edge(Dudinka, Tchita)
add_edge(Dudinka, Mongolia)
add_edge(Siberia, Tchita)
add_edge(Siberia, Vladvostok)
add_edge(Tchita, Vladvostok)
add_edge(Tchita, Mongolia)
add_edge(Tchita, China)
add_edge(Vladvostok, China)
add_edge(Vladvostok, Japao)
add_edge(Aral, Oriente_Medio)
add_edge(Aral, India)
add_edge(Aral, China)
add_edge(Mongolia, China)
add_edge(China, Japao)
add_edge(China, Vietna)
add_edge(China, India)
add_edge(Vietna, India)
add_edge(India, Oriente_Medio)

# África
add_edge(Argelia, Egito)
add_edge(Argelia, Sudao)
add_edge(Argelia, Congo)
add_edge(Egito, Sudao)
add_edge(Sudao, Congo)
add_edge(Sudao, Madagascar)
add_edge(Sudao, Africa_do_Sul)
add_edge(Congo, Africa_do_Sul)
add_edge(Africa_do_Sul, Madagascar)

# Oceania
add_edge(Sumatra, Australia)
add_edge(Borneu, Nova_Guine)
add_edge(Borneu, Australia)
add_edge(Nova_Guine, Australia)

# Intercontinentais
# América do Sul - América do Norte
add_edge(Venezuela, Mexico)

# América do Sul - África
add_edge(Brasil, Argelia)

# América do Norte - Europa
add_edge(Groelandia, Islandia)

# América do Norte - Asia
add_edge(Alasca, Vladvostok)

# Europa - Asia
add_edge(Polonia, Oriente_Medio)
add_edge(Moscou, Omsk)
add_edge(Moscou, Aral)
add_edge(Moscou, Oriente_Medio)

# Europa - África
add_edge(Franca, Argelia)
add_edge(Franca, Egito)
add_edge(Polonia, Egito)

# Asia - África
add_edge(Oriente_Medio, Egito)

# Asia - Oceania
add_edge(India, Sumatra)
add_edge(Vietna, Borneu)

def verify_conquered_continents(player):
    conquered_continents = []
    for continent in continents:
        if set(continent) <= set(player.territories):
            conquered_continents.append(continent)

    return conquered_continents

def continent_to_troops(continent):
    if Mexico in continent:
        return 5
    if Brasil in continent:
        return 2
    if Franca in continent:
        return 5
    if Argelia in continent:
        return 4
    if India in continent:
        return 7
    if Australia in continent:
        return 2
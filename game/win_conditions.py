# Conquistar na totalidade a EUROPA, a board.continents["oceania"] e mais um terceiro.
def objective_one(player, colors, board):
    if (set(board.continents["europe"]) < set(player.territories) and 
        set(board.continents["oceania"]) < set(player.territories)):
        if (set(board.continents["south_america"]) < set(player.territories) or 
            set(board.continents["north_america"]) < set(player.territories) or 
            set(board.continents["asia"]) < set(player.territories) or 
            set(board.continents["africa"]) < set(player.territories)):
            return True
    return False

# Conquistar na totalidade a ÁSIA e a AMÉRICA DO SUL.
def objective_two(player, colors, board):
    if (set(board.continents["south_america"]) < set(player.territories) and 
        set(board.continents["asia"]) < set(player.territories)):
        return True
    return False

# Conquistar na totalidade a EUROPA, a AMÉRICA DO SUL e mais um terceiro.
def objective_three(player, colors, board):
    if (set(board.continents["europe"]) < set(player.territories) and 
        set(board.continents["south_america"]) < set(player.territories)):
        if (set(board.continents["oceania"]) < set(player.territories) or 
            set(board.continents["north_america"]) < set(player.territories) or 
            set(board.continents["asia"]) < set(player.territories) or 
            set(board.continents["africa"]) < set(player.territories)):
            return True
    return False

# Conquistar 18 TERRITÓRIOS e ocupar cada um deles com pelo menos dois exércitos.
def objective_four(player, colors, board):
    if (len(player.territories) >= 18):
        territories_2 = 0
        for i in player.territories:
            if i.troops >= 2:
                territories_2 += 1
            if territories_2 >= 18:
                return True
    return False

# Conquistar na totalidade a ÁSIA e a ÁFRICA.
def objective_five(player, colors, board):
    if (set(board.continents["asia"]) < set(player.territories) and 
        set(board.continents["africa"]) < set(player.territories)):
        return True
    return False

# Conquistar na totalidade a AMÉRICA DO NORTE e a ÁFRICA.
def objective_six(player, colors, board):
    if (set(board.continents["north_america"]) < set(player.territories) and 
        set(board.continents["africa"]) < set(player.territories)):
        return True
    return False

# Conquistar 24 TERRITÓRIOS à sua escolha.
def objective_seven(player, colors, board):
    if (len(player.territories) >= 24):
        return True
    return False

# Conquistar na totalidade a AMÉRICA DO NORTE e a board.continents["oceania"].
def objective_eight(player, colors, board):
    if (set(board.continents["north_america"]) < set(player.territories) and set(board.continents["oceania"]) < set(player.territories)):
        return True
    return False


def objective_nine(player, colors, board):
    if 'Azul' not in colors or player.color == 'Azul':
        return len(player.territories) >= 24
    
    if len(colors['Azul'].territories) == 0:
        return True



def objective_ten(player, colors, board):
    if 'Amarelo' not in colors or player.color == 'Amarelo':
        return len(player.territories) >= 24
    
    if len(colors['Amarelo'].territories) == 0:
        return True


def objective_eleven(player, colors, board):
    if 'Vermelho' not in colors or player.color == 'Vermelho':
        return len(player.territories) >= 24
    
    if len(colors['Vermelho'].territories) == 0:
        return True


def objective_twelve(player, colors, board):
    if 'Cinza' not in colors or player.color == 'Cinza':
        return len(player.territories) >= 24
    
    if len(colors['Cinza'].territories) == 0:
        return True


def objective_thirteen(player, colors, board):
    if 'Roxo' not in colors or player.color == 'Roxo':
        return len(player.territories) >= 24
    
    if len(colors['Roxo'].territories) == 0:
        return True


def objective_fourteen(player, colors, board):
    if 'Verde' not in colors or player.color == 'Verde':
        return len(player.territories) >= 24
    
    if len(colors['Verde'].territories) == 0:
        return True


objectives = [objective_one,
              objective_two,
              objective_three,
              objective_four,
              objective_five,
              objective_six,
              objective_seven,
              objective_eight,
              objective_nine,
              objective_ten,
              objective_eleven,
              objective_twelve,
              objective_thirteen,
              objective_fourteen]

objectives_descriptions = {
    objective_one: 'Conquer all of board.continents["europe"], board.continents["oceania"], and another continent of your choice.',
    objective_two: 'Conquer all of SOUTH AMERICA and board.continents["asia"].',
    objective_three: 'Conquer all of board.continents["europe"], SOUTH AMERICA, and another continent of your choice.',
    objective_four: 'Conquer 18 TERRITORIES and occupy each one with at least two armies.',
    objective_five: 'Conquer all of board.continents["asia"] and board.continents["africa"].',
    objective_six: 'Conquer all of NORTH AMERICA and board.continents["africa"].',
    objective_seven: 'Conquer 24 TERRITORIES of your choice.',
    objective_eight: 'Conquer all of NORTH AMERICA and board.continents["oceania"].',
    objective_nine: 'Eliminate the BLUE player. If you are the BLUE player or the BLUE player does not exist or the BLUE player is eliminated by someone else, then your objective is to conquer 24 territories.',
    objective_ten: 'Eliminate the YELLOW player. If you are the YELLOW player or the YELLOW player does not exist or the YELLOW player is eliminated by someone else, then your objective is to conquer 24 territories.',
    objective_eleven: 'Eliminate the RED player. If you are the RED player or the RED player does not exist or the RED player is eliminated by someone else, then your objective is to conquer 24 territories.',
    objective_twelve: 'Eliminate the GRAY player. If you are the GRAY player or the GRAY player does not exist or the GRAY player is eliminated by someone else, then your objective is to conquer 24 territories.',
    objective_thirteen: 'Eliminate the PURPLE player. If you are the PURPLE player or the PURPLE player has no territories and is still alive, or the PURPLE player is eliminated by someone else, then your objective is to conquer 24 territories.',
    objective_fourteen: 'Eliminate the GREEN player. If you are the GREEN player or the GREEN player has no territories and is still alive, or the GREEN player is eliminated by someone else, then your objective is to conquer 24 territories.'
}


def check_win(current_player, players, board):
    colors = {}
    for player in players:
        # Add all players alive at the beggining of the round
        if not player.has_died:
            colors[player.name] = player.color
        # Mark player as dead if necessary, after adding him to this turn's elimination list
        if len(player.territories) == 0:
            player.has_died = True

    if current_player.objective(current_player, colors, board):
        return True
    return False

def simple_check_win(current_player, players, board):
    alive_players = []
    for player in players:
        if not player.has_died:
            alive_players.append(player)
        # Mark player as dead if necessary
        if len(player.territories) == 0:
            player.has_died = True
        

    if len(alive_players) == 1:
        return True
    return False
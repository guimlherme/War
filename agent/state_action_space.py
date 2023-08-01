from game.territories import Board

board = Board()

territory_ids = range(len(board.territories_data))  # List of territory identifiers

# Define the action space
action_space = []
action_space += [-1] # finish phase
action_space += list(territory_ids)

# Define the dimension of the state space
len_state_space = 1 + 2 * len(board.territories_data)
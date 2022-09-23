import numpy as np
import math

from simple_xo.game import SimpleXOGame

DEFAULT_PARENT_VISITS = 1e-8

def features_for_board_and_player(board, player):
    features = np.zeros((3, 3))
    features[board == player] = 1
    return features

def get_features(game_state):
    current_player = game_state[1]
    opponent = 3 - current_player
    features = [
        features_for_board_and_player(game_state[0], current_player),
        features_for_board_and_player(game_state[0], opponent)
    ]
    return np.array(features)

def edge_key(features, chosen_move):
    return (features.tobytes(), np.array(chosen_move).tobytes())

def node_key(features):
    return features.tobytes()

def game_state_from_game(game: SimpleXOGame):
    return (np.array(game.board), game.player)

def update_state(game_state, move):
    game = SimpleXOGame(game_state)
    game.play_current_player(move)
    return game_state_from_game(game)

def get_valid_moves_from_array(valid_moves_array):
    return np.flip(np.argwhere(valid_moves_array))

def get_probabilities_array(agent_policy, valid_moves_array):
    valid_moves = np.ma.masked_array(agent_policy, ~valid_moves_array)
    return valid_moves / np.ma.sum(valid_moves)

# C_UCT to override the one used in sim
def C_UCT(move, exploration, game_state, valid_moves, N, Q):
    edge = edge_key(get_features(game_state), move)
    assert edge in Q

    uct = (
        exploration
        * (
            math.sqrt(
                sum(
                    N[edge_key(get_features(game_state), valid_move)]
                    for valid_move in valid_moves
                )
                + DEFAULT_PARENT_VISITS
            )
            / (1 + N[edge])
        )
        + Q[edge]
    )
    return uct
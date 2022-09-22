import numpy as np
import numpy.typing as npt
from .game import XOGame

import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})
from cython_modules.cython_test import UCT as C_UCT
from cython_modules.cython_test import get_features as C_get_features


def edge_key(features, chosen_move):
    return (features.tobytes(), chosen_move.tobytes())


def node_key(features):
    return features.tobytes()


def game_state_from_game(game: XOGame):
    return (np.array(game.board), game.player, game._last_move)


def update_state(game_state, move):
    game = XOGame(game_state)
    game.play_current_player(move)
    return game_state_from_game(game)


def get_probabilities_array(agent_policy, valid_moves_array) -> npt.NDArray:
    valid_moves = np.ma.masked_array(agent_policy, ~valid_moves_array)
    return valid_moves / np.ma.sum(valid_moves)


def get_valid_moves_from_array(valid_moves_array):
    return np.flip(np.argwhere(valid_moves_array))


def UCT(
    move,
    exploration,
    game_state,
    valid_moves,
    N,
    Q,
):
    return C_UCT(move, exploration, game_state, valid_moves, N, Q)

def get_features(game_state):
    return C_get_features(game_state)

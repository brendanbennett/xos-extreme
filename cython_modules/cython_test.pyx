import math
import numpy as np
cimport numpy as np
np.import_array()
import cython

DTYPE_int = np.int

ctypedef np.int_t DTYPE_int_t
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef inline np.ndarray[DTYPE_int_t, ndim=2] features_for_board_and_player(np.ndarray[DTYPE_int_t, ndim=2] board, int player):
    cdef np.ndarray[DTYPE_int_t, ndim=2] features = np.zeros((9, 9), dtype=DTYPE_int)
    # features[features != player] = 0
    for y in range(9):
        for x in range(9):
            if board[y, x] == player:
                features[y, x] = 1
    # features[features != 0] = 1
    return features


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[DTYPE_int_t, ndim=3] get_features(np.ndarray[DTYPE_int_t, ndim=2] game_state_0, int game_state_1, game_state_2):
    cdef int current_player = int(game_state_1)
    cdef int opponent = int(3 - current_player)  # mafs lol
    cdef np.ndarray[DTYPE_int_t, ndim=2] last_play = np.zeros((9, 9), dtype=DTYPE_int)

    if game_state_2 is not None:
        last_play[game_state_2[1], game_state_2[0]] = 1
    features = [
        features_for_board_and_player(game_state_0, current_player),
        features_for_board_and_player(game_state_0, opponent),
        last_play,
    ]
    return np.array(features)

def edge_key(game_state, chosen_move):
    return (get_features(game_state[0].astype(DTYPE_int), game_state[1], game_state[2]).tobytes(), chosen_move.tobytes())

def UCT(move, exploration, game_state, valid_moves, N, Q):
    DEFAULT_PARENT_VISITS = 1e-8

    edge = edge_key(game_state, move)

    assert edge in Q

    uct = exploration * (
        math.sqrt(sum(N[edge_key(game_state, valid_move)] for valid_move in valid_moves)+DEFAULT_PARENT_VISITS)
        / (1 + N[edge])
    ) + Q[edge]
    return uct

import pickle

def profile():
    with open("cython_modules/sample_uct_input.obj", "rb") as f:
        data = pickle.load(f)
    for i in range(1000):
        UCT(*data.values())
    return
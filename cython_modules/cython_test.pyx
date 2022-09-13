import math
import numpy as np
cimport numpy as np
np.import_array()

import random
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
cdef np.ndarray[DTYPE_int_t, ndim=3] get_features(game_state):
    cdef int current_player = game_state[1]
    cdef int opponent = 3 - current_player  # mafs lol
    cdef np.ndarray[DTYPE_int_t, ndim=2] last_play = np.zeros((9, 9), dtype=DTYPE_int)

    if game_state[2] is not None:
        last_play[game_state[2][1], game_state[2][0]] = 1
    features = [
        features_for_board_and_player(game_state[0].astype(DTYPE_int), current_player),
        features_for_board_and_player(game_state[0].astype(DTYPE_int), opponent),
        last_play,
    ]
    return np.array(features, dtype=DTYPE_int)

def edge_key(game_state, chosen_move):
    return (get_features(game_state).tobytes(), chosen_move.tobytes())

def UCT(move, exploration, game_state, valid_moves, N, Q):
    DEFAULT_PARENT_VISITS = 1e-8 * random.uniform(0.5,1)

    edge = edge_key(game_state, move)

    assert edge in Q

    uct = exploration * (
        math.sqrt(sum(N[edge_key(game_state, valid_move)] for valid_move in valid_moves)+DEFAULT_PARENT_VISITS)
        / (1 + N[edge])
    ) + Q[edge]
    return uct

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int C_check_win_from_scratch(board):
    cdef np.ndarray[DTYPE_int_t, ndim=2] new_board = np.zeros((9, 9), dtype=DTYPE_int)
    cdef int sum1
    cdef int sum2
    cdef int e
    for y in range(3):
        for x in range(3):
            e = board[y, x]
            if e == 1:
                new_board[y, x] = -1
            elif e == 2:
                new_board[y, x] = 1

    for i in range(3):
        sum1 = np.sum(new_board[i,:])
        sum2 = np.sum(new_board[:,i])
        if sum1 == 3 or sum2 == 3:
            return 2
        elif sum1 == -3 or sum2 == -3:
            return 1
    sum1 = np.sum(new_board.diagonal())
    sum2 = np.sum(np.fliplr(new_board).diagonal())
    if sum1 == 3 or sum2 == 3:
        return 2
    elif sum1 == -3 or sum2 == -3:
        return 1
    return -1

def check_win_from_scratch(board):
    return C_check_win_from_scratch(board)

import pickle

def profile():
    with open("cython_modules/sample_uct_input.obj", "rb") as f:
        data = pickle.load(f)
    for i in range(1000):
        UCT(*data.values())
    return

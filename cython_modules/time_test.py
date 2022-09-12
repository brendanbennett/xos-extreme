import numpy as np
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})
from test_copy import get_features


def get_features_py(board):
    features = board
    features[0, 0] = 5
    return board


board_p = get_features_py(np.arange(9).reshape((3, 3)))
board_c = get_features(np.arange(9).reshape((3, 3)))
print(board_p)
print(board_c)

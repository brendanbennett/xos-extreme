import numpy as np
cimport numpy as np
np.import_array()

def get_features(board):
    cdef np.ndarray features
    features = np.array(board)
    features[0,0] = 5
    return board

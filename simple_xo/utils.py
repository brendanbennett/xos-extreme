import numpy as np

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
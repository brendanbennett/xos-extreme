from ..game import XOGame
import numpy as np

def test_check_winning_play():
    win_board = np.array([2,1,1,0,2,1,1,0,2]).reshape((3,3))
    assert XOGame._check_for_xo_winning_play(win_board, (1, 1)) == 2



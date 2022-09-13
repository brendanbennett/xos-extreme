import timeit
from game.game import XOGame
import numpy as np

# One self play makes about 480000 calls which takes 7/30s (here is worst case where it takes 15s)
def test_check_win_time(n = 480000):
    board = np.array([0,1,2,1,0,2,0,2,1]).reshape((3,3))
    # print(timeit.timeit(lambda: game.check_win_from_scratch(board), number=n))
    print(timeit.timeit(lambda: XOGame.check_win_from_scratch_beta(board), number=n))

if __name__ == "__main__":
    test_check_win_time()
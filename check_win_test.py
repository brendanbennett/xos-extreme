from timeit import timeit

import numpy as np
import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
from cython_modules.cython_test import check_win_from_scratch as C_check_win_from_scratch

def generate_all_boards():
    all_boards_strs = [f"{np.base_repr(i, 3):0>9}" for i in range(3**9)]
    all_boards = [np.array([int(i) for i in list(k)], dtype=np.int8).reshape(3, 3) for k in all_boards_strs]
    return all_boards

def index_from_board(board):
    return int("".join([str(int(i)) for i in board.flatten()]), 3)

def get_winners(boards):
    winners = [C_check_win_from_scratch(board) for board in boards]
    winners = [k if k != -1 else 0 for k in winners]
    return winners

def test_boards_idx():
    all_boards = generate_all_boards()
    for i, board in enumerate(all_boards):
        assert index_from_board(board) == i
        
winners = get_winners(generate_all_boards())
        
def check_win_lookup(board):
    return winners[index_from_board(board)]
    
def main():
    all_boards = generate_all_boards()
    def time_lookup():
        for board in all_boards:
            check_win_lookup(board)
            
    def time_old():
        for board in all_boards:
            C_check_win_from_scratch(board)
    
    print(timeit(time_lookup, number=10))
    print(timeit(time_old, number=10))
    
if __name__ == "__main__":
    main()
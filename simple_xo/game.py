import numpy as np

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
from cython_modules.cython_test import check_win_from_scratch as C_check_win_from_scratch

# game_state is (board, player to play)

class SimpleXOGame:
    def __init__(self, game_state=None) -> None:
        self.board = np.zeros((3,3))
        self.player: int = 1
        self.winner: int | None = None

        self._valid_moves_array = np.full((3, 3), True)
        if game_state is not None:
            self._init_from_game_state(game_state)

    def _init_from_game_state(self, game_state):
        self.board = game_state[0]
        self.player = game_state[1]
        self._valid_moves_array, self.winner = self.valid_moves_array_and_winner_from_state(game_state)

    @staticmethod
    def _check_win_from_scratch(board):
        
        wc = C_check_win_from_scratch(board)
        wc = wc if wc != -1 else None
        if wc is None and not np.any(board == 0):
            return 0
        return wc

    @staticmethod
    def valid_moves_array_and_winner_from_state(game_state):
        board, player = game_state
        winner = SimpleXOGame._check_win_from_scratch(board)
        valid_moves_array = board == 0
        return valid_moves_array, winner

    def _set_board(self, coords: tuple[int, int], value: int) -> None:
        self.board[coords[1], coords[0]] = value

    def _get_board(self, coords: tuple[int, int]) -> int:
        return self.board[coords[1], coords[0]]

    def _iterate_players(self) -> None:
        if self.player == 1:
            self.player = 2
        else:
            self.player = 1

    def _update_win_state(self) -> None:
        self.winner = SimpleXOGame._check_win_from_scratch(self.board)

    def _update_valid_moves_array(self) -> None:
        self._valid_moves_array = self.board == 0

    def _is_valid(self, coords: tuple[int, int]) -> bool:
        val = self._get_board(coords)
        return True if val == 0 else False

    def get_valid_moves(self):
        return np.fliplr(np.argwhere(self._valid_moves_array))

    def play_current_player(self, coords: tuple[int, int]) -> None:
        assert self._is_valid(coords)
        self._set_board(coords, self.player)

        self._update_win_state()
        self._update_valid_moves_array()
        self._iterate_players()

if __name__ == "__main__":
    rng = np.random.RandomState()

    game = SimpleXOGame()
    while game.winner is None:
        valid_moves_indices = game.get_valid_moves()
        random_choice = rng.randint(len(valid_moves_indices), size=1)[0]
        chosen_move = valid_moves_indices[random_choice]
        game.play_current_player(chosen_move)
    print(game.board)
    print(game.winner)
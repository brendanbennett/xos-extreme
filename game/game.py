import numpy as np
import numpy.typing as npt
from copy import deepcopy
import time
from collections import Counter


class XOGame:
    def __init__(self) -> None:
        self.board = np.zeros((9, 9))
        # players 1 and 2. 1 goes first
        self.player: int = 1
        self.winner: int = None

        self._large_board = np.zeros((3, 3))
        self._last_move: tuple[int, int] = None
        self._valid_moves = np.full((9, 9), True)

    def _set_board(self, coords: tuple[int, int], value: int) -> None:
        self.board[coords[1], coords[0]] = value

    def _get_board(self, coords: tuple[int, int]) -> int:
        return self.board[coords[1], coords[0]]

    def _set_large_board(self, coords: tuple[int, int], value: int) -> None:
        self._large_board[coords[1], coords[0]] = value

    def _get_large_board(self, coords: tuple[int, int]) -> int:
        return self._large_board[coords[1], coords[0]]

    def _get_small_board(self, large_coords: tuple[int, int]):
        """Return the small board corresponding to the large board coordinates"""
        return self.board[
            3 * large_coords[1] : 3 * large_coords[1] + 3,
            3 * large_coords[0] : 3 * large_coords[0] + 3,
        ]

    @staticmethod
    def _get_large_coords(small_coords: tuple[int, int]) -> tuple[int, int]:
        return (small_coords[0] // 3, small_coords[1] // 3)

    @staticmethod
    def _get_small_board_coords(small_coords: tuple[int, int]) -> tuple[int, int]:
        return (small_coords[0] % 3, small_coords[1] % 3)

    @staticmethod
    def _get_large_coords_for_next(small_coords: tuple[int, int]) -> tuple[int, int]:
        return (small_coords[0] % 3, small_coords[1] % 3)

    def _iterate_players(self) -> None:
        if self.player == 1:
            self.player = 2
        else:
            self.player = 1

    def _update_valid_moves(self) -> None:
        """Return valid moves as one hot boolean np array"""
        large_coords = self._get_large_coords_for_next(self._last_move)
        if self._get_large_board(large_coords) == 0:
            send_to = np.zeros((3, 3))
            send_to[large_coords[1], large_coords[0]] = 1
            mask = np.repeat(np.repeat(send_to, 3, 0), 3, 1)
            masked = np.logical_and(self.board == 0, mask)
            self._valid_moves = masked
        else:
            mask = np.repeat(np.repeat(self._large_board, 3, 0), 3, 1)
            self._valid_moves = 0 == self.board + mask
        if not np.any(self._valid_moves) and self.winner is None:
            self.winner = 0

    @staticmethod
    def _check_for_xo_winning_play(board, coords_of_last_play: tuple[int, int]) -> int:
        """Check for winning play of normal game of xo"""
        this_player_board = deepcopy(board)
        player = this_player_board[coords_of_last_play[1], coords_of_last_play[0]]
        this_player_board[board != player] = 0
        if any(
            (
                np.all(this_player_board[coords_of_last_play[1], :]),
                np.all(this_player_board[:, coords_of_last_play[0]]),
            )
        ):
            return player
        elif coords_of_last_play[0] == coords_of_last_play[1] and np.all(
            this_player_board.diagonal()
        ):
            return player
        elif coords_of_last_play[0] + coords_of_last_play[1] == 2 and np.all(
            np.fliplr(this_player_board).diagonal()
        ):
            return player
        else:
            return 0

    def _update_win_state(self) -> None:
        # Check if small board is full
        large_coords = self._get_large_coords(self._last_move)
        small_board = self._get_small_board(large_coords)
        small_board_coords = self._get_small_board_coords(self._last_move)
        small_board_win = self._check_for_xo_winning_play(
            small_board, small_board_coords
        )
        if small_board_win:
            # print(f"Player {small_board_win} wins {large_coords}")
            self._set_large_board(large_coords, small_board_win)
            large_board_win = self._check_for_xo_winning_play(
                self._large_board, large_coords
            )
            if large_board_win:
                # print("large:")
                # print(self._large_board)
                # print("small:")
                # print(small_board)
                self.winner = large_board_win

    def is_valid(self, coords) -> bool:
        return self._valid_moves[coords[1], coords[0]]

    def get_valid_moves(self):
        return np.fliplr(np.argwhere(self._valid_moves))

    def play_current_player(self, coords: tuple[int, int]) -> None:
        assert self.is_valid(coords)
        self._set_board(coords, self.player)
        self._last_move = coords
        self._update_win_state()

        self._update_valid_moves()
        self._iterate_players()

    def display_board(self) -> None:
        print(self.board)


if __name__ == "__main__":
    rng = np.random.RandomState(100)
    
    winners = []

    start = time.perf_counter()
    for n in range(1000):
        game = XOGame()
        for i in range(81):
            # print(game._valid_moves)

            valid_moves_indices = game.get_valid_moves()
            # print(f"valid moves {valid_moves_indices}")
            random_choice = rng.randint(len(valid_moves_indices), size=1)[0]
            chosen_move = valid_moves_indices[random_choice]
            # print(f"{game.player} playing {chosen_move}")
            game.play_current_player(chosen_move)
            # game.display_board()
            if game.winner is not None:
                # print(f"Player {game.winner} wins!")
                # print(game._large_board)
                break
        winners.append(game.winner)
    end = time.perf_counter()

    player_wins = Counter(winners)
    print(player_wins)
    print(f"took {end - start} seconds.")

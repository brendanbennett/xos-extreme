from ..game import XOGame
import numpy as np
from collections import Counter
from copy import deepcopy


def game_state_from_game(game: XOGame):
    return (deepcopy(game.board), game.player, game._last_move)


def test_check_winning_play():
    win_board = np.array([2, 1, 1, 0, 2, 1, 1, 0, 2]).reshape((3, 3))
    assert XOGame._check_for_xo_winning_play(win_board, (1, 1)) == 2


def test_check_win():
    win_2_board = np.array([2, 1, 1, 0, 2, 1, 1, 0, 2]).reshape((3, 3))
    win_1_board = np.array([1, 1, 1, 0, 2, 1, 1, 0, 2]).reshape((3, 3))
    win_none_board = np.array([0, 1, 1, 0, 2, 1, 1, 0, 2]).reshape((3, 3))
    win_anti_diag_board = np.array([0, 0, 1, 0, 1, 0, 1, 0, 0]).reshape((3, 3))
    assert XOGame.check_win_from_scratch(win_2_board) == 2
    assert XOGame.check_win_from_scratch(win_1_board) == 1
    assert XOGame.check_win_from_scratch(win_none_board) == None
    assert XOGame.check_win_from_scratch(win_anti_diag_board) == 1


def test_game():
    rng = np.random.RandomState(100)

    winners = []

    for n in range(100):
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

    player_wins = dict(Counter(winners))
    assert player_wins == {0: 29, 1.0: 36, 2.0: 35}


def test_load_game_state():
    game = XOGame()
    game.play_current_player((8, 8))
    init_game_state = game_state_from_game(game)
    game.play_current_player((7, 8))
    game_state1 = game_state_from_game(game)

    game2 = XOGame(init_game_state)
    game2.play_current_player((7, 8))
    game_state2 = game_state_from_game(game2)
    assert np.all(game_state1[0] == game_state2[0])
    assert game_state1[1] == game_state2[1]
    assert np.all(game_state1[2] == game_state2[2])

from game.game import XOGame
from copy import deepcopy
import numpy as np
import numpy.typing as npt
from collections import defaultdict
import math
import time
from pathlib import Path
import pickle
from agent import XOAgentBase, XOAgentModel, Network
from re import compile, fullmatch
from torch import load

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
from cython_modules.cython_test import UCT as C_UCT
from cython_modules.cython_test import get_features

TRAINING_DATA_RE = compile("training_data_\d+_\d+\.obj")

def features_for_board_and_player(board, player):
    features = np.zeros((9, 9))
    features[board == player] = 1
    return features


def edge_key(features, chosen_move):
    return (features.tobytes(), chosen_move.tobytes())


def node_key(features):
    return features.tobytes()


def game_state_from_game(game: XOGame):
    return (np.array(game.board), game.player, game._last_move)


def update_state(game_state, move):
    game = XOGame(game_state)
    game.play_current_player(move)
    return game_state_from_game(game)


def get_probabilities_array(agent_policy, valid_moves_array) -> npt.NDArray:
    valid_moves = np.ma.masked_array(agent_policy, ~valid_moves_array)
    return valid_moves / np.ma.sum(valid_moves)


def get_valid_moves_from_array(valid_moves_array):
    return np.flip(np.argwhere(valid_moves_array))

class MCTS:
    def __init__(self) -> None:
        self.Q = defaultdict(float)
        self.N = defaultdict(int)
        self.W = defaultdict(float)
        self.P = dict()  # is always created with a value
        self.trajectories = (
            dict()
        )  # keys are features as bytes, values are list of moves
        self.exploration = 1

        self.timings = defaultdict(list)

    def _get_edges_from_parent(self, game_state) -> list:
        """Returns a list of edge keys originating from a game state"""
        valid_moves_array, winner = XOGame.valid_moves_array_and_winner_from_state(
            game_state
        )
        valid_moves = get_valid_moves_from_array(valid_moves_array)
        return [edge_key(get_features(game_state), valid_move) for valid_move in valid_moves]

    def _total_parent_N(self, game_state) -> list:
        edges = self._get_edges_from_parent(game_state)
        return [self.N[edge] for edge in edges]

    def _uct_select(self, game_state):
        DEFAULT_PARENT_VISITS = 1e-8

        valid_moves_array, winner = XOGame.valid_moves_array_and_winner_from_state(
            game_state
        )
        valid_moves = get_valid_moves_from_array(valid_moves_array)

        # start = time.perf_counter()
        assert all(edge_key(get_features(game_state), move) in self.P for move in valid_moves)
        # end = time.perf_counter()
        # self.timings["assert"].append(end - start)

        # Modified from alphago zero paper to add a small number to uct so it's never zero.
        def UCT(move):
            edge = edge_key(get_features(game_state), move)
            assert edge in self.Q

            uct = (
                self.exploration
                * (
                    math.sqrt(
                        sum(
                            self.N[edge_key(get_features(game_state), valid_move)]
                            for valid_move in valid_moves
                        )
                        + DEFAULT_PARENT_VISITS
                    )
                    / (1 + self.N[edge])
                )
                + self.Q[edge]
            )
            return uct

        def UCT_import(
            move,
            exploration=self.exploration,
            game_state=game_state,
            valid_moves=valid_moves,
            N=self.N,
            Q=self.Q,
        ):
            return C_UCT(move, exploration, game_state, valid_moves, N, Q)

        # start = time.perf_counter()
        # print(list(map(lambda m : round(UCT_import(m), 2), valid_moves)))
        # print(list(map(lambda m : round(UCT(m), 2), valid_moves)))
        # print("\n")
        # selected = max(valid_moves, key=lambda m : C_UCT(m, self.exploration, game_state, valid_moves, self.N, self.Q))
        selected = max(valid_moves, key=UCT_import)
        # end = time.perf_counter()
        # self.timings["max"].append(end - start)

        return selected

    def _is_expanded(self, features):
        return node_key(features) in self.trajectories

    def _add_trajectory(self, node_key, move):
        try:
            self.trajectories[node_key].append(move)
        except KeyError:
            self.trajectories[node_key] = [move]

    def _get_trajectories(self, game_state):
        try:
            return self.trajectories[node_key(get_features(game_state))]
        except KeyError:
            return []

    def _select(self, game_state):
        visited_edges = []
        active_game_state = deepcopy(game_state)
        features = get_features(active_game_state)
        while True:
            if not self._is_expanded(features) or not self._get_trajectories(
                active_game_state
            ):
                # reason = "not expanded" if not self._is_expanded(active_game_state) else "empty trajectories"
                # print(f"Found unexpanded game state because {reason}")

                # print(active_game_state)
                return active_game_state, visited_edges
            # start = time.perf_counter()
            move = self._uct_select(active_game_state)
            # end = time.perf_counter()
            # self.timings["uct_select"].append(end - start)
            # print(f"Selected move {move} with uct")
            visited_edges.append(edge_key(features, move))
            # print("game state before update:")
            # print(active_game_state)
            active_game_state = update_state(active_game_state, move)
            features = get_features(active_game_state)
            # print("Next game state after uct:")
            # print(active_game_state)

    def _expand(self, game_state, agent: XOAgentBase) -> float:
        """Returns value evaluated by agent"""
        features = get_features(game_state)
        agent_policy, value = agent.get_policy_and_value(features)

        valid_moves_array = XOGame.valid_moves_array_and_winner_from_state(game_state)[
            0
        ]
        valid_moves = get_valid_moves_from_array(valid_moves_array)
        if len(valid_moves) != 0:
            probabilities = get_probabilities_array(agent_policy, valid_moves_array)
            for move in valid_moves:
                # print(f"Adding move {move} to trajectories")
                edge = edge_key(features, move)
                self.Q[edge], self.N[edge], self.W[edge] = 0, 0, 0
                self.P[edge] = probabilities[move[1], move[0]]
                self._add_trajectory(node_key(features), move)
        return value

    def _backup(self, visited_edges: list, reward):
        # print(f"backup {len(visited_edges)} edges")
        for edge in reversed(visited_edges):
            self.N[edge] += 1
            self.W[edge] += reward
            self.Q[edge] = self.W[edge] / self.N[edge]
            reward = 1 - reward

    def probabilities_for_state(self, game_state, temp):
        valid_moves_array, winner = XOGame.valid_moves_array_and_winner_from_state(
            game_state
        )
        valid_moves = get_valid_moves_from_array(valid_moves_array)

        # Shouldn't be terminal
        assert winner is None

        root_visits = sum(self._total_parent_N(game_state))

        # Also this shouldn't be called without running any searches
        assert root_visits > 0
        valid_moves_probabilities = [
            self.N[edge_key(get_features(game_state), move)] / root_visits for move in valid_moves
        ]
        probabilities_array = np.zeros((9, 9))
        probabilities_array[tuple(np.fliplr(valid_moves).T)] = valid_moves_probabilities
        probabilities_array = np.power(probabilities_array, 1 / temp)
        probabilities_array = probabilities_array / np.sum(probabilities_array)
        return probabilities_array

    def rollout(self, game_state, agent: XOAgentBase):
        # start = time.perf_counter()
        leaf_game_state, visited_edges = self._select(game_state)
        # end = time.perf_counter()
        # self.timings["select"].append(end - start)
        # print("Trajectories before expand:")
        # print(self.trajectories.values())
        # start = time.perf_counter()
        value = self._expand(leaf_game_state, agent)
        # end = time.perf_counter()
        # self.timings["expand"].append(end - start)
        # print("Trajectories after expand:")
        # print(self.trajectories.values())
        # start = time.perf_counter()
        self._backup(visited_edges, value)
        # end = time.perf_counter()
        # self.timings["backup"].append(end - start)

        # print("\n\n")

    def select_new_parent(self, game_state, move):
        features = get_features(game_state)
        edges_to_remove = self._get_edges_from_parent(game_state)
        edges_to_remove.remove(edge_key(features, move))
        for edge in edges_to_remove:
            map(lambda x: x.pop(edge), (self.Q, self.N, self.W, self.P))

        self.trajectories.pop(node_key(features), None)

    def self_play(self, agent, rollouts_per_move=200):
        game = XOGame()
        training_states = []
        for j in range(81):
            # print(f"Move {j+1}")
            game_state = game_state_from_game(game)
            for i in range(rollouts_per_move):
                # if i % 20 == 0:
                #     print(f"{i/rollouts_per_move*100:.0f}%")
                self.rollout(game_state, agent)

            probabilities = self.probabilities_for_state(game_state, 1)
            chosen_move = np.flip(
                np.unravel_index(np.argmax(probabilities), shape=(9, 9))
            )
            training_states.append([get_features(game_state), probabilities])

            game.play_current_player(chosen_move)

            if game.winner is not None:
                print(f"Player {game.winner} wins!")
                # print(game._large_board)
                if game.winner == 1:
                    reward = 1
                elif game.winner == 2:
                    reward = -1
                else:
                    reward = 0
                for i in range(len(training_states)):
                    training_states[i].append(reward)
                    reward *= -1
                break

            self.select_new_parent(game_state, chosen_move)

        return training_states


def profiling():
    net = Network(2 + 1, 32, 4)
    agent = XOAgentModel(net)
    game_state = game_state_from_game(XOGame())

    monte = MCTS()
    rollouts = 200
    for i in range(rollouts):
        monte.rollout(game_state, agent)


def save_training_data(
    list_of_self_play_games,
    agent_revision: int,
    chunk_size=32,
    training_dir="training_data",
):
    revisions_paths = [x for x in Path(".", training_dir).iterdir() if x.is_dir()]
    revision_path = Path(".", training_dir, str(agent_revision))
    if revision_path not in revisions_paths:
        revision_path.mkdir()

    chunk_paths = [
        x for x in revision_path.iterdir() if x.is_file() and fullmatch(TRAINING_DATA_RE, x.name)
    ]
    chunk_idxs = [
        (int(path.stem.split("_")[-2]), int(path.stem.split("_")[-1]))
        for path in chunk_paths
    ]

    new_chunk_start_offset = (
        0 if len(chunk_idxs) == 0 else sorted(chunk_idxs, key=lambda i: i[1])[-1][1] + 1
    )

    for chunk_start in range(0, len(list_of_self_play_games), chunk_size):
        try:
            chunk = list_of_self_play_games[chunk_start : chunk_start + chunk_size]
        except KeyError:
            chunk = list_of_self_play_games[chunk_start:]
        name = f"training_data_{new_chunk_start_offset + chunk_start}_{new_chunk_start_offset + chunk_start + len(chunk) - 1}.obj"
        assert fullmatch(TRAINING_DATA_RE, name)
        print(f"Saving {name}")
        with open(Path(revision_path, name), "wb") as f:
            pickle.dump(chunk, f)

def evaluate_agents(agent1: XOAgentBase, agent2: XOAgentBase, games: int = 40, rollouts_per_move: int = 200):


    winners = {0: 0, 1: 0, 2: 0}
    # agent1 starts as player 1
    for game_num in range(games):
        MCTS_1 = MCTS()
        MCTS_2 = MCTS()
        game = XOGame()
        first_player_agent = 1
        for i in range(81):
            current_agent_idx = (first_player_agent + i + 1) % 2 + 1
            current_MCTS = MCTS_1 if current_agent_idx == 1 else MCTS_2
            opponent_MCTS = MCTS_2 if current_agent_idx == 1 else MCTS_1
            current_agent = agent1 if current_agent_idx == 1 else agent2
            game_state = game_state_from_game(game)
            for i in range(rollouts_per_move):
                current_MCTS.rollout(game_state, current_agent)

            probabilities = current_MCTS.probabilities_for_state(game_state, 1)
            chosen_move = np.flip(
                np.unravel_index(np.argmax(probabilities), shape=(9, 9))
            )

            game.play_current_player(chosen_move)

            if game.winner is not None:
                winning_agent_idx = (first_player_agent + game.winner) % 2 + 1
                winning_agent_idx = 0 if game.winner == 0 else int(winning_agent_idx)
                print(f"Agent {winning_agent_idx} wins game {game_num + 1}")
                winners[winning_agent_idx] += 1
                break
            
            current_MCTS.select_new_parent(game_state, chosen_move)
            opponent_MCTS.select_new_parent(game_state, chosen_move)
        first_player_agent = 2 - first_player_agent
    return winners


def main():
    # feature planes of current board state (first for current player, second for next)
    # and one hot encoding of last move
    # net = Network(2 + 1, 32, 4)
    # agent = XOAgentModel(net)

    # monte = MCTS()
    
    # while True:
    #     list_training_data = []
    #     for i in range(8):
    #         list_training_data.append(monte.self_play(agent, 200))
    #     save_training_data(list_training_data, 0)

    trained_model = Network()
    trained_model.load_state_dict(load("models/trained_model_4"))
    trained_agent = XOAgentModel(trained_model)

    evaluate_agents(XOAgentModel(Network()), trained_agent, games=10, rollouts_per_move=200)
    breakpoint()

if __name__ == "__main__":
    main()

from game.game import XOGame
import torch.nn as nn
import torch
from torchsummary import summary
from collections import deque
from copy import deepcopy
import numpy as np


class Convolutional(nn.Module):
    def __init__(self, feature_planes: int, conv_filters: int) -> None:
        super().__init__()

        self.conv = nn.Conv2d(feature_planes, conv_filters, 3, 1, padding=1)
        self.batch_norm = nn.BatchNorm2d(conv_filters)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        output = torch.relu(x)
        return output


class Residual(nn.Module):
    def __init__(self, n_filters) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(n_filters, n_filters, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(n_filters)

        self.conv2 = nn.Conv2d(n_filters, n_filters, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(n_filters)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = torch.relu(out)

        return out


class PolicyHead(nn.Module):
    def __init__(self, n_filters) -> None:
        super().__init__()
        self.conv = nn.Conv2d(n_filters, 2, 1, 1)
        self.bn = nn.BatchNorm2d(2)

        self.linear = nn.Linear(2 * 9 * 9, 9 * 9)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = torch.relu(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        out = torch.sigmoid(out)
        # Shape (batch_num, 81)

        return out


class ValueHead(nn.Module):
    def __init__(self, n_filters, hidden) -> None:
        super().__init__()
        self.conv = nn.Conv2d(n_filters, 1, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        # relu
        self.linear1 = nn.Linear(9 * 9, hidden)
        # relu
        self.linear2 = nn.Linear(hidden, 1)
        # tanh

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = torch.relu(out)
        out = torch.flatten(out, 1)
        out = self.linear1(out)
        out = torch.relu(out)
        out = self.linear2(out)
        out = torch.tanh(out)
        # Shape (batch_num, 1)

        return out


class Network(nn.Module):
    def __init__(
        self, feature_planes: int, conv_filters: int, n_residuals: int
    ) -> None:
        super().__init__()

        self.convolutional = Convolutional(feature_planes, conv_filters)

        self.residuals = nn.Sequential(
            *[Residual(conv_filters) for _ in range(n_residuals)]
        )

        self.policy_head = PolicyHead(conv_filters)

        self.value_head = ValueHead(conv_filters, conv_filters)

    def forward(self, x):
        out = self.convolutional(x)
        out = self.residuals(out)

        policy_out = self.policy_head(out)
        value_out = self.value_head(out)

        # Shape (batch_num, 81+1)
        output = torch.cat((policy_out, value_out), dim=1)

        return output


class XOAgentBase:
    def get_policy_and_value(self, features) -> tuple:
        pass


class XOAgentModel(XOAgentBase):
    def __init__(self, model: nn.Module) -> None:
        self.model = model.float()

    def get_policy_and_value(self, features):
        self.model.eval()
        model_out = self.model(features)
        return torch.split(model_out, [81, 1], dim=1)


class XOAgentRandom(XOAgentBase):
    def __init__(self, seed) -> None:
        self.rng = np.random.RandomState(seed)

    def get_policy_and_value(self, features) -> tuple:
        # valid_moves_indices = game.get_valid_moves()
        # # print(f"valid moves {valid_moves_indices}")
        # random_choice = self.rng.randint(len(valid_moves_indices), size=1)[0]
        # chosen_move = valid_moves_indices[random_choice]
        # # print(f"{game.player} playing {chosen_move}")
        pass


def features_for_board_and_player(board, player):
    features = deepcopy(board)
    features[features != player] = 0
    features[features != 0] = 1
    return features


def get_features(board_states: deque):
    current_player = board_states[-1][1]
    opponent = board_states[-2][1]
    features = []
    for i in range(1, 3):
        features.append(
            features_for_board_and_player(board_states[-i][0], current_player)
        )
        features.append(features_for_board_and_player(board_states[-i][0], opponent))
    features_tensor = torch.from_numpy(np.array(features)[np.newaxis])
    features_tensor = features_tensor.to(torch.float32)
    return features_tensor


def self_play(agent: XOAgentBase):
    game = XOGame()

    game_states = deque()
    game_states.append((game.board, 2))
    game_states.append((game.board, 1))

    for i in range(81):
        features = get_features(game_states)

        agent_policy, agent_value = agent.get_policy_and_value(features)
        # print(agent_policy, agent_value)

        agent_policy = agent_policy.cpu().detach().numpy().reshape((9, 9))
        valid_moves = np.ma.masked_array(agent_policy, ~game._valid_moves)
        chosen_move = np.flip(np.unravel_index(valid_moves.argmax(fill_value=-1), (9, 9)))

        # print(chosen_move)

        game.play_current_player(chosen_move)
        game_states.popleft()
        game_states.append((game.board, game.player))
        if game.winner is not None:
            break

    print(f"Player {game.winner} wins!")


if __name__ == "__main__":
    # feature planes of last 2 turns of 2 players
    net = Network(2 * 2, 32, 4)

    # policy = PolicyHead(32)

    # summary(policy, (32, 1, 1))

    # summary(net, (4, 9, 9))

    agent = XOAgentModel(net)

    # input = rand((1, 4, 9, 9))

    # output = agent.get_policy_and_value(input)

    # print(output)

    self_play(agent)

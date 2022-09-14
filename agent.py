import torch
import torch.nn as nn
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
        self, feature_planes: int = 3, conv_filters: int = 32, n_residuals: int = 4
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

        features_tensor = torch.from_numpy(np.array(features)[np.newaxis])
        features_tensor = features_tensor.to(torch.float32)
        model_out = self.model(features_tensor)
        output_policy, output_value = torch.split(model_out, [81, 1], dim=1)
        return output_policy.cpu().detach().numpy().reshape((9, 9)), output_value.item()

    @staticmethod
    def policy_and_value_to_model_out(policy, value):
        return torch.from_numpy(np.concatenate((policy.flatten(), [value]))[np.newaxis]).to(torch.float32)


class XOAgentRandom(XOAgentBase):
    def __init__(self, seed) -> None:
        self.rng = np.random.RandomState(seed)

    def get_policy_and_value(self, features) -> tuple:
        return torch.from_numpy(self.rng.rand(9, 9)), None
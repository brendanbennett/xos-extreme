import torch
import torch.nn as nn
import numpy as np
import torchsummary


class Convolutional(nn.Module):
    def __init__(self, feature_planes: int, conv_filters: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            feature_planes, conv_filters, 3, stride=3, padding=0, groups=feature_planes
        )
        self.batch_norm1 = nn.BatchNorm2d(conv_filters)

        self.conv2 = nn.Conv2d(
            conv_filters, conv_filters, 3, stride=1, padding=1
        )
        self.batch_norm2 = nn.BatchNorm2d(conv_filters)
        
        self.conv3 = nn.Conv2d(
            conv_filters, conv_filters, 3, stride=1, padding=1
        )
        self.batch_norm3 = nn.BatchNorm2d(conv_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = torch.relu(x)
        return x


class PolicyHead(nn.Module):
    def __init__(self, conv_filters, board_edge_len) -> None:
        super().__init__()
        self.conv = nn.Conv2d(conv_filters, 2, 1, stride=1, padding=0)
        self.batch_norm = nn.BatchNorm2d(2)
        self.linear = nn.Linear(2*3*3, (board_edge_len**2))

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = torch.softmax(x, dim=1)
        # Shape (batch_num, 81)

        return x


class ValueHead(nn.Module):
    def __init__(self, conv_filters, hidden) -> None:
        super().__init__()
        self.conv = nn.Conv2d(conv_filters, 1, 1, stride=1, padding=0)
        self.batch_norm = nn.BatchNorm2d(1)
        self.linear1 = nn.Linear(3*3, hidden)
        self.linear2 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.tanh(x)
        # Shape (batch_num, 1)

        return x


class Network(nn.Module):
    def __init__(
        self,
        feature_planes: int = 3,
        conv_filters: int = 36,
        hidden: int = 64,
        board_edge_len: int = 9,
    ) -> None:
        super().__init__()

        self.convolutional = Convolutional(feature_planes, conv_filters)

        self.policy_head = PolicyHead(
            conv_filters, board_edge_len
        )

        self.value_head = ValueHead(
            conv_filters, hidden
        )

    def forward(self, x):
        out = self.convolutional(x)

        policy_out = self.policy_head(out)
        value_out = self.value_head(out)

        # Shape (batch_num, 81+1)
        output = torch.cat((policy_out, value_out), dim=1)

        return output
    
    
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor):
        x = torch.relu(self.linear1(x.view(x.size(0), -1)))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        x = torch.cat((torch.softmax(x[:, :-1], dim=1), torch.tanh(x[:, -1:])), dim=1)
        return x


class XOAgentBase:
    def get_policy_and_value(self, features) -> tuple:
        pass


class XOAgentModel(XOAgentBase):
    def __init__(
        self, model: nn.Module = None, feature_planes=3, board_edge_len=9
    ) -> None:
        self.feature_planes = feature_planes
        self.board_edge_len = board_edge_len
        if model is not None:
            self.model = model.float()
        else:
            self.model = Network(
                feature_planes=feature_planes, board_edge_len=board_edge_len
            )

    def get_policy_and_value(self, features):
        self.model.eval()

        features_tensor = torch.from_numpy(np.array(features)[np.newaxis])
        features_tensor = features_tensor.to(torch.float32)
        model_out = self.model(features_tensor)
        output_policy, output_value = torch.split(
            model_out, [self.board_edge_len**2, 1], dim=1
        )
        return (
            output_policy.cpu()
            .detach()
            .numpy()
            .reshape((self.board_edge_len, self.board_edge_len)),
            output_value.item(),
        )

    @staticmethod
    def policy_and_value_to_model_out(policy, value):
        return torch.from_numpy(
            np.concatenate((policy.flatten(), [value]))[np.newaxis]
        ).to(torch.float32)


class XOAgentRandom(XOAgentBase):
    def __init__(self, seed) -> None:
        self.rng = np.random.RandomState(seed)

    def get_policy_and_value(self, features) -> tuple:
        return torch.from_numpy(self.rng.rand(9, 9)), None
    

if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    net = Network()
    net.to("cuda")
    conv = Convolutional(3)
    conv.to("cuda")
    torchsummary.summary(net, (3, 9, 9))

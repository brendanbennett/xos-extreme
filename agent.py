import torch
import torch.nn as nn
import numpy as np


class Convolutional(nn.Module):
    def __init__(self, feature_planes: int, conv_filters: int = 32, head_hidden: int = 128) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(feature_planes, conv_filters, 3, stride=3, padding=0) 
        self.batch_norm1 = nn.BatchNorm2d(conv_filters)
        
        self.conv2 = nn.Conv2d(conv_filters, head_hidden, 3, stride=1, padding=0)
        self.batch_norm2 = nn.BatchNorm2d(head_hidden)
        self.flatten = nn.Flatten(start_dim=1) # Don't flatten batches

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.flatten(x)
        output = torch.relu(x)
        return output

class PolicyHead(nn.Module):
    def __init__(self, head_hidden, board_edge_len) -> None:
        super().__init__()

        self.linear = nn.Linear(head_hidden, (board_edge_len**2))

    def forward(self, x):
        x = self.linear(x)
        out = torch.sigmoid(x)
        # Shape (batch_num, 81)

        return out


class ValueHead(nn.Module):
    def __init__(self, head_hidden, hidden, board_edge_len) -> None:
        super().__init__()
        
        self.linear1 = nn.Linear(head_hidden, hidden)
        self.linear2 = nn.Linear(hidden, 1)
        
    def forward(self, x):
        out = self.linear1(x)
        out = torch.relu(out)
        out = self.linear2(out)
        out = torch.tanh(out)
        # Shape (batch_num, 1)

        return out


class Network(nn.Module):
    def __init__(
        self,
        feature_planes: int = 3,
        conv_filters: int = 32,
        head_hidden: int = 128,
        hidden: int = 64,
        board_edge_len: int = 9,
    ) -> None:
        super().__init__()

        self.convolutional = Convolutional(feature_planes, conv_filters)

        self.policy_head = PolicyHead(head_hidden, board_edge_len=board_edge_len)

        self.value_head = ValueHead(head_hidden, hidden, board_edge_len=board_edge_len)

    def forward(self, x):
        out = self.convolutional(x)

        policy_out = self.policy_head(out)
        value_out = self.value_head(out)

        # Shape (batch_num, 81+1)
        output = torch.cat((policy_out, value_out), dim=1)

        return output


class XOAgentBase:
    def get_policy_and_value(self, features) -> tuple:
        pass


class XOAgentModel(XOAgentBase):
    def __init__(self, model: nn.Module = None, feature_planes=3, board_edge_len=9) -> None:
        self.feature_planes = feature_planes
        self.board_edge_len = board_edge_len
        if model is not None:
            self.model = model.float()
        else:
            self.model = Network(feature_planes=feature_planes, board_edge_len=board_edge_len)

    def get_policy_and_value(self, features):
        self.model.eval()

        features_tensor = torch.from_numpy(np.array(features)[np.newaxis])
        features_tensor = features_tensor.to(torch.float32)
        model_out = self.model(features_tensor)
        output_policy, output_value = torch.split(model_out, [self.board_edge_len**2, 1], dim=1)
        return output_policy.cpu().detach().numpy().reshape((self.board_edge_len, self.board_edge_len)), output_value.item()

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

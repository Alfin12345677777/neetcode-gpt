import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        std = math.sqrt(2 / (fan_in + fan_out))

        torch.manual_seed(0)
        randomw = torch.randn(fan_out, fan_in) * std

        rounded = torch.round(randomw, decimals=4)

        return rounded.tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        std = math.sqrt(2 / fan_in)

        torch.manual_seed(0)
        randomw = torch.randn(fan_out, fan_in) * std

        rounded = torch.round(randomw, decimals=4)

        return rounded.tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        torch.manual_seed(0)

        weights = []

        for layer in range(num_layers):
            if layer == 0:
                fan_in = input_dim
            else:
                fan_in = hidden_dim

            fan_out = hidden_dim

            if init_type == "xavier":
                std = math.sqrt(2 / (fan_in + fan_out))
            elif init_type == "kaiming":
                std = math.sqrt(2 / fan_in)
            elif init_type == "random":
                std = 1.0

            W = torch.randn(fan_out, fan_in) * std
            weights.append(W)

        x = torch.randn(input_dim)

        stds = []

        for W in weights:
            x = x @ W.T
            x = torch.relu(x)
            stds.append(round(x.std().item(), 2))

        return stds
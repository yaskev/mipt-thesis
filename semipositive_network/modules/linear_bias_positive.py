import torch
from torch import Tensor


class LinearBiasPositive(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(LinearBiasPositive, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(in_features, out_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))

    def forward(self, input: Tensor) -> Tensor:
        return input @ torch.exp(self.weight) + torch.exp(self.bias)

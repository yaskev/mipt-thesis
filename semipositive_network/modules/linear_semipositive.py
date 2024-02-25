import torch
from torch import Tensor


class LinearSemiPositive(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # self.moneyness_weight = torch.nn.Parameter(torch.randn(out_features))
        # self.weight = torch.nn.Parameter(torch.randn(in_features - 1, out_features))
        self.weight = torch.nn.Parameter(torch.randn(in_features, out_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))

    def forward(self, input: Tensor) -> Tensor:
        # semi_input = torch.clone(input)
        #     semi_input[:, 0] = 1 / semi_input[:, 0]
        #     semi_input[:, 2] = 0.5 - semi_input[:, 2]
        # return semi_input[:, 1:] @ torch.exp(self.weight) + self.bias + torch.outer(semi_input[:, 0], self.moneyness_weight)
        return input @ torch.exp(self.weight) + self.bias

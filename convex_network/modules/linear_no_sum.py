import torch
from torch import Tensor


class LinearNoSum(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(in_features, out_features))
        self.bias = torch.nn.Parameter(torch.randn(in_features, out_features))
        self.in_feats = in_features
        self.out_feats = out_features

    def forward(self, input: Tensor) -> Tensor:
        batch_size = input.size()[0]
        input = input.repeat_interleave(self.bias.size()[1], 0)
        input = torch.transpose(input, 0, 1)
        res = torch.zeros((self.in_feats, self.out_feats * batch_size))
        for i in range(batch_size):
            res[:, i * self.out_feats: (i + 1) * self.out_feats]\
                = input[:, i * self.out_feats: (i + 1) * self.out_feats] * torch.exp(self.weight) + self.bias
        return res

    def backward(self, grad_output: Tensor) -> Tensor:
        return grad_output * torch.exp(self.weight)

import torch
from torch import Tensor


class SoftSigmoid(torch.nn.Module):
    def __init__(self, convex_indices: Tensor, output_dim: int):
        super().__init__()
        self.convex_indices = convex_indices
        self.output_dim = output_dim

    def forward(self, input: Tensor) -> Tensor:
        rows, cols = input.size()[0], input.size()[1]
        for i in range(rows):
            if i in self.convex_indices:
                # Hand-written softplus
                input[i, :] = torch.log(1 + torch.exp(input[i, :]))
            else:
                input[i, :] = torch.sigmoid(input[i, :])
        return input.prod(dim=0).reshape((int(cols / self.output_dim), self.output_dim))

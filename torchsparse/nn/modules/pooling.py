import torch
from torch import nn

from torchsparse import SparseTensor
from torchsparse.nn import functional as F

__all__ = ['GlobalAvgPool', 'GlobalMaxPool']


class GlobalAvgPool(nn.Module):

    def forward(self, input: SparseTensor) -> torch.Tensor:
        return F.global_avg_pool(input)


class GlobalMaxPool(nn.Module):

    def forward(self, input: SparseTensor) -> torch.Tensor:
        return F.global_max_pool(input)

from torch import nn

from torchsparse import SparseTensor
from torchsparse.nn import functional as F

__all__ = ['GlobalAveragePooling', 'GlobalMaxPooling']


class GlobalAveragePooling(nn.Module):

    def forward(self, input: SparseTensor) -> SparseTensor:
        return F.global_avg_pool(input)


class GlobalMaxPooling(nn.Module):

    def forward(self, input: SparseTensor) -> SparseTensor:
        return F.global_max_pool(input)

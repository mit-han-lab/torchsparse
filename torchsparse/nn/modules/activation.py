from torch import nn

from torchsparse import SparseTensor
from torchsparse.nn.utils import fapply

__all__ = ['ReLU', 'LeakyReLU']


class ReLU(nn.ReLU):

    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)


class LeakyReLU(nn.LeakyReLU):

    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)

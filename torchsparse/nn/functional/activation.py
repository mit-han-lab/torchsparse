from torch.nn import functional as F

from torchsparse import SparseTensor
from torchsparse.nn.utils import fapply

__all__ = ["relu", "silu", "leaky_relu"]


def relu(input: SparseTensor, inplace: bool = True) -> SparseTensor:
    return fapply(input, F.relu, inplace=inplace)


def silu(input: SparseTensor, inplace: bool = True) -> SparseTensor:
    return fapply(input, F.silu, inplace=inplace)


def leaky_relu(
    input: SparseTensor, negative_slope: float = 0.1, inplace: bool = True
) -> SparseTensor:
    return fapply(input, F.leaky_relu, negative_slope=negative_slope, inplace=inplace)

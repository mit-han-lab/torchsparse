import functools

from torch import nn
from torchsparse.sparse_tensor import *

from ..functional import *

__all__ = ['ReLU', 'LeakyReLU']


class Activation(nn.Module):
    def __init__(self, inplace: bool = True) -> None:
        super().__init__()
        self.activation = spact
        self.inplace = inplace

    def forward(self, inputs):
        return self.activation(inputs)


class ReLU(Activation):
    def __init__(self, inplace: bool = True) -> None:
        super().__init__()
        self.activation = functools.partial(sprelu, inplace=inplace)

    def __repr__(self):
        if self.inplace:
            return 'ReLU(inplace=True)'
        else:
            return 'ReLU(inplace=False)'


class LeakyReLU(Activation):
    def __init__(self, negative_slope: float = 0.1,
                 inplace: bool = True) -> None:
        super().__init__()
        self.activation = functools.partial(spleaky_relu,
                                            negative_slope=negative_slope,
                                            inplace=inplace)
        self.negative_slope = negative_slope

    def __repr__(self):
        if self.inplace:
            return 'LeakyReLU(negative_slope=%f, inplace=True)' % self.negative_slope
        else:
            return 'LeakyReLU(negative_slope=%f, inplace=False)' % self.negative_slope

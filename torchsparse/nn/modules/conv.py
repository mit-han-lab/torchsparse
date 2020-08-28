import math

import torch
from torch import nn

from torchsparse.sparse_tensor import *

from ..functional import *

__all__ = ['Conv3d']


class Conv3d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 dilation: int = 1,
                 bias: bool = False,
                 transpose: bool = False) -> None:
        super().__init__()
        self.in_channels = inc = in_channels
        self.out_channels = outc = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.kernel = nn.Parameter(
            torch.zeros(self.kernel_size ** 3, inc,
                        outc)) if self.kernel_size > 1 else nn.Parameter(
                            torch.zeros(inc, outc))
        self.bias = None if not bias else nn.Parameter(torch.zeros(outc))
        self.t = transpose
        self.init_weight()

        if kernel_size == 1:
            assert not transpose

    def __repr__(self):
        if not self.t:
            return 'Conv3d(in_channels=%d, out_channels=%d, kernel_size=%d, stride=%d, dilation=%d)' % (
                self.in_channels, self.out_channels, self.kernel_size,
                self.stride, self.dilation)
        else:
            return 'Conv3d(in_channels=%d, out_channels=%d, kernel_size=%d, stride=%d, dilation=%d)' % (
                self.in_channels, self.out_channels, self.kernel_size,
                self.stride, self.dilation)

    def init_weight(self):
        std = 1. / math.sqrt(
            self.out_channels if self.t else self.in_channels *
            (self.kernel_size ** 3))
        self.kernel.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, inputs):
        return conv3d(inputs,
                      self.kernel,
                      self.bias,
                      stride=self.stride,
                      dilation=self.dilation,
                      transpose=self.t)

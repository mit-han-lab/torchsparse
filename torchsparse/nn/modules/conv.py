import math
import sys
from typing import Dict, List, Tuple, Union

if sys.version_info >= (3, 8):
    from functools import cached_property
else:
    from backports.cached_property import cached_property

import numpy as np
import torch
from torch import nn

import torchsparse
from torchsparse import SparseTensor
from torchsparse.nn import functional as F
from torchsparse.utils import make_ntuple

__all__ = ["Conv3d"]


class Conv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]] = 3,
        stride: Union[int, List[int], Tuple[int, ...]] = 1,
        padding: Union[int, Tuple[int, ...]] = 0,
        dilation: int = 1,
        bias: bool = False,
        transposed: bool = False,
        generative: bool = False,
        config: Dict = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = make_ntuple(kernel_size, ndim=3)
        self.stride = make_ntuple(stride, ndim=3)
        self.dilation = dilation
        _padding = make_ntuple(padding, 3)
        self.padding = ()
        for i in range(3):
            if self.kernel_size[i] % 2 == 1 and self.stride[i] == 1:
                self.padding += ((self.kernel_size[i] - 1) // 2,)
            else:
                self.padding += (_padding[i],)
        self.transposed = transposed
        self.generative = generative
        if self.generative:
            assert self.transposed

        self._config = config

        self.kernel_volume = int(np.prod(self.kernel_size))
        if (
            self.kernel_volume > 1
            or self.kernel_volume == 1
            and self.stride != (1, 1, 1)
        ):
            self.kernel = nn.Parameter(
                torch.zeros(self.kernel_volume, in_channels, out_channels)
            )
        else:
            self.kernel = nn.Parameter(torch.zeros(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def extra_repr(self) -> str:
        s = "{in_channels}, {out_channels}, kernel_size={kernel_size}"
        if self.stride != (1,) * len(self.stride):
            s += ", stride={stride}"
        if self.dilation != 1:
            s += ", dilation={dilation}"
        if self.bias is None:
            s += ", bias=False"
        if self.transposed:
            s += ", transposed=True"
        if self.generative:
            s += ", generative=True"
        return s.format(**self.__dict__)

    def reset_parameters(self) -> None:
        std = 1 / math.sqrt(
            (self.out_channels if self.transposed else self.in_channels)
            * self.kernel_volume
        )
        self.kernel.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, input: SparseTensor) -> SparseTensor:

        return F.conv3d(
            input,
            weight=self.kernel,
            kernel_size=self.kernel_size,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            transposed=self.transposed,
            generative=self.generative,
            config=self._config,
            training=self.training,
        )

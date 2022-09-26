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

__all__ = ['Conv3d']


class Conv3d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]] = 3,
                 stride: Union[int, List[int], Tuple[int, ...]] = 1,
                 dilation: int = 1,
                 bias: bool = False,
                 transposed: bool = False,
                 config: Dict = None) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = make_ntuple(kernel_size, ndim=3)
        self.stride = make_ntuple(stride, ndim=3)
        self.dilation = dilation
        self.transposed = transposed

        if config is None:
            config = {}
        config['epsilon'] = config.get('epsilon', 0.0)
        config['mm_thresh'] = config.get('mm_thresh', 0)
        config['kmap_mode'] = config.get('kmap_mode', 'hashmap')
        self.config = config

        self.kernel_volume = int(np.prod(self.kernel_size))
        if self.kernel_volume > 1:
            self.kernel = nn.Parameter(
                torch.zeros(self.kernel_volume, in_channels, out_channels))
        else:
            self.kernel = nn.Parameter(torch.zeros(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def extra_repr(self) -> str:
        s = '{in_channels}, {out_channels}, kernel_size={kernel_size}'
        if self.stride != (1,) * len(self.stride):
            s += ', stride={stride}'
        if self.dilation != 1:
            s += ', dilation={dilation}'
        if self.bias is None:
            s += ', bias=False'
        if self.transposed:
            s += ', transposed=True'
        return s.format(**self.__dict__)

    def reset_parameters(self) -> None:
        std = 1 / math.sqrt(
            (self.out_channels if self.transposed else self.in_channels)
            * self.kernel_volume)
        self.kernel.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    @cached_property
    def _reordered_kernel(self) -> nn.Parameter:
        kernel_data = torch.zeros_like(self.kernel.data)
        ind = 0
        while ind < self.kernel_volume - 1:
            kernel_data[ind] = self.kernel.data[ind // 2].clone()
            kernel_data[ind + 1] = \
                self.kernel.data[self.kernel_volume - 1 - ind // 2].clone()
            ind += 2
        if self.kernel_volume % 2 == 1:
            kernel_data[self.kernel_volume - 1] = \
                self.kernel.data[self.kernel_volume // 2].clone()
        return nn.Parameter(kernel_data, requires_grad=False)

    def forward(self, input: SparseTensor) -> SparseTensor:
        kernel = self.kernel
        epsilon, mm_thresh = self.config['epsilon'], self.config['mm_thresh']
        if torchsparse.backends.benchmark:  # type: ignore
            if self.training:
                print('Warning: it is not recommended to enable '
                      + 'torchsparse.backends.benchmark during the training.')
                epsilon, mm_thresh = 0.0, 0
            elif (self.config['epsilon'] != 0.0
                    or self.config['mm_thresh'] != 0) and \
                    len(kernel.data.shape) == 3:
                kernel = self._reordered_kernel

        return F.conv3d(input,
                        kernel,
                        kernel_size=self.kernel_size,
                        bias=self.bias,
                        stride=self.stride,
                        dilation=self.dilation,
                        transposed=self.transposed,
                        epsilon=epsilon,
                        mm_thresh=mm_thresh,
                        kmap_mode=self.config['kmap_mode'])

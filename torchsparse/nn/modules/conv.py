import math
from typing import Tuple, Union

import numpy as np
import torch
from torch import nn

from torchsparse import SparseTensor
from torchsparse.nn import functional as F
from torchsparse.utils import make_ntuple

__all__ = ['Conv3d']


class Conv3d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, ...]] = 3,
                 stride: Union[int, Tuple[int, ...]] = 1,
                 dilation: int = 1,
                 bias: bool = False,
                 transposed: bool = False,
                 config=None) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = make_ntuple(kernel_size, ndim=3)
        self.stride = make_ntuple(stride, ndim=3)
        self.dilation = dilation
        self.transposed = transposed

        # if config == None:
        #     config = defaultdict(int)
        #     if kernel_size == 3:
        #         config = {'epsilon': 0.4, 'mm_thresh': 20000}
        #     elif kernel_size == 2:
        #         config = {'epsilon': 1.0, 'mm_thresh': 20000}
        if config is None:
            config = {}
        config['epsilon'] = config.get('epsilon', 0.0)
        config['mm_thresh'] = config.get('mm_thresh', 0)
        config['kmap_mode'] = config.get('kmap_mode', 'hashmap')
        config['conv_mode'] = config.get('conv_mode', 'default')
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

    def forward(self, input: SparseTensor) -> SparseTensor:
        return F.conv3d(input,
                        self.kernel,
                        kernel_size=self.kernel_size,
                        bias=self.bias,
                        stride=self.stride,
                        dilation=self.dilation,
                        transposed=self.transposed,
                        epsilon=self.config['epsilon'],
                        mm_thresh=self.config['mm_thresh'],
                        kmap_mode=self.config['kmap_mode'],
                        conv_mode=self.config['conv_mode'])

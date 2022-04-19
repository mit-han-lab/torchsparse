from typing import Union

from torch import nn

from torchsparse import nn as spnn
from torchsparse.tensor import SparseTensor
from torchsparse.utils import make_ntuple

__all__ = ['SparseConvBlock', 'SparseConvTransposeBlock', 'SparseResBlock']


class SparseConvBlock(nn.Sequential):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, list, tuple],
                 stride: Union[int, list, tuple] = 1,
                 dilation: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        dilation=dilation),
            spnn.BatchNorm(out_channels),
            spnn.ReLU(True),
        )

    def forward(self, x: SparseTensor):
        return self.net(x)


class SparseConvTransposeBlock(nn.Sequential):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, list, tuple],
                 stride: Union[int, list, tuple] = 1,
                 dilation: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        dilation=dilation,
                        transposed=True),
            spnn.BatchNorm(out_channels),
            spnn.ReLU(True),
        )

    def forward(self, x: SparseTensor):
        return self.net(x)


class SparseResBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, list, tuple],
                 stride: Union[int, list, tuple] = 1,
                 dilation: int = 1) -> None:
        super().__init__()
        self.main = nn.Sequential(
            spnn.Conv3d(in_channels,
                        out_channels,
                        kernel_size,
                        dilation=dilation,
                        stride=stride),
            spnn.BatchNorm(out_channels),
            spnn.ReLU(True),
            spnn.Conv3d(out_channels,
                        out_channels,
                        kernel_size,
                        dilation=dilation),
            spnn.BatchNorm(out_channels),
        )

        if in_channels != out_channels or any(
                x > 1 for x in make_ntuple(stride, ndim=3)):
            self.shortcut = nn.Sequential(
                spnn.Conv3d(in_channels, out_channels, 1, stride=stride),
                spnn.BatchNorm(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = spnn.ReLU(True)

    def forward(self, x: SparseTensor):
        x = self.relu(self.main(x) + self.shortcut(x))
        return x

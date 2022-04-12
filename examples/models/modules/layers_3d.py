from torch import nn

from torchsparse import nn as spnn

__all__ = ['SparseConvBlock', 'SparseDeConvBlock', 'SparseResBlock']


class SparseConvBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 dilation: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        dilation=dilation), spnn.BatchNorm(out_channels),
            spnn.ReLU(True))

    def forward(self, x):
        return self.net(x)


class SparseDeConvBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 dilation: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        dilation=dilation,
                        transposed=True), spnn.BatchNorm(out_channels),
            spnn.ReLU(True))

    def forward(self, x):
        return self.net(x)


class SparseResBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 dilation: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(in_channels,
                        out_channels,
                        kernel_size,
                        dilation=dilation,
                        stride=stride), spnn.BatchNorm(out_channels),
            spnn.ReLU(True),
            spnn.Conv3d(out_channels,
                        out_channels,
                        kernel_size,
                        dilation=dilation), spnn.BatchNorm(out_channels))

        if in_channels != out_channels or stride > 1:
            self.downsample = nn.Sequential(
                spnn.Conv3d(in_channels, out_channels, 1, stride=stride),
                spnn.BatchNorm(out_channels),
            )
        else:
            self.downsample = nn.Sequential()

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        x = self.relu(self.net(x) + self.downsample(x))
        return x

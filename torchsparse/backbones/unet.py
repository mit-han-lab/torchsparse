import torch.nn as nn

import torchsparse
import torchsparse.nn as spnn
from torchsparse.backbones.modules import (SparseConvBlock,
                                           SparseConvTransposeBlock,
                                           SparseResBlock)
from torchsparse.tensor import SparseTensor

__all__ = ['SparseResUNet18']


class SparseResUNet18(nn.Module):

    def __init__(self,
                 in_channels: int = 4,
                 channel_sizes: list = None,
                 width_multiplier: float = 1.0):
        super().__init__()

        cs = channel_sizes if channel_sizes is not None else [
            32, 32, 64, 128, 256, 256, 128, 96, 96
        ]
        cs = [int(width_multiplier * x) for x in cs]

        self.stem = nn.Sequential(
            spnn.Conv3d(in_channels, cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]),
            spnn.ReLU(True),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]),
            spnn.ReLU(True),
        )

        self.downs = []
        for i in range(4):
            self.downs.append(
                nn.Sequential(
                    SparseConvBlock(cs[i],
                                    cs[i],
                                    kernel_size=2,
                                    stride=2,
                                    dilation=1),
                    SparseResBlock(cs[i],
                                   cs[i + 1],
                                   kernel_size=3,
                                   stride=1,
                                   dilation=1),
                    SparseResBlock(cs[i + 1],
                                   cs[i + 1],
                                   kernel_size=3,
                                   stride=1,
                                   dilation=1),
                ))
        self.downs = nn.ModuleList(self.downs)

        self.ups = []
        for i in range(4):
            self.ups.append(
                nn.ModuleList([
                    SparseConvTransposeBlock(cs[i + 4],
                                             cs[i + 5],
                                             kernel_size=2,
                                             stride=2),
                    nn.Sequential(
                        SparseResBlock(cs[i + 5] + cs[3 - i],
                                       cs[i + 5],
                                       kernel_size=3,
                                       stride=1,
                                       dilation=1),
                        SparseResBlock(cs[i + 5],
                                       cs[i + 5],
                                       kernel_size=3,
                                       stride=1,
                                       dilation=1),
                    )
                ]))
        self.ups = nn.ModuleList(self.ups)

    def forward(self, x: SparseTensor):
        feats = {}
        down_keys = ['down1', 'down2', 'down3', 'down4']
        up_keys = ['up1', 'up2', 'up3', 'out']

        feats['stem'] = self.stem(x)

        for i in range(4):
            if i == 0:
                feats[down_keys[i]] = self.downs[i](feats['stem'])
            else:
                feats[down_keys[i]] = self.downs[i](feats[down_keys[i - 1]])

        for i in range(4):
            if i == 0:
                x = self.ups[i][0](feats[down_keys[-1]])
            else:
                x = self.ups[i][0](feats[up_keys[i - 1]])

            if i == 3:
                x = torchsparse.cat([x, feats['stem']])
            else:
                x = torchsparse.cat([x, feats[down_keys[2 - i]]])

            feats[up_keys[i]] = self.ups[i][1](x)

        return feats

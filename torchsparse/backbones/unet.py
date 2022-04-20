from typing import Dict, List, Optional

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
                 channel_sizes: Optional[List[int]] = None,
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

        self.encoders = nn.ModuleList()
        for i in range(4):
            self.encoders.append(
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

        self.decoders = nn.ModuleList()
        for i in range(4):
            self.decoders.append(
                nn.ModuleDict({
                    'upsample':
                        SparseConvTransposeBlock(cs[i + 4],
                                                 cs[i + 5],
                                                 kernel_size=2,
                                                 stride=2),
                    'fuse':
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
                }))

    def forward(self, x: SparseTensor) -> Dict[str, SparseTensor]:

        def dfs(
            x: SparseTensor,
            encoders: nn.ModuleList,
            decoders: nn.ModuleList,
        ) -> List[SparseTensor]:
            if not encoders and not decoders:
                return [x]

            # downsample
            xd = encoders[0](x)

            # inner recursion
            outputs = dfs(xd, encoders[1:], decoders[:-1])
            yd = outputs[-1]

            # upsample and fuse
            u = decoders[-1]['upsample'](yd)
            y = decoders[-1]['fuse'](torchsparse.cat([u, x]))

            return [x] + outputs + [y]

        output_dict = {}
        output_dict['stem'] = x = self.stem(x)
        outputs = dfs(x, self.encoders, self.decoders)
        for k, name in enumerate(
            ['down1', 'down2', 'down3', 'down4', 'up1', 'up2', 'up3', 'out'],
                1):
            output_dict[name] = outputs[k]
        return output_dict

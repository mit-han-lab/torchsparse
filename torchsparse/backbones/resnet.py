from typing import List, Tuple, Union

from torch import nn

from torchsparse import SparseTensor

from .modules import SparseConvBlock, SparseResBlock

__all__ = ['SparseResNet21D']


class SparseResNet(nn.ModuleList):

    def __init__(
        self,
        blocks: List[Tuple[int, int, Union[int, Tuple[int, ...]],
                           Union[int, Tuple[int, ...]]]],
        *,
        in_channels: int = 4,
        width_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.blocks = blocks
        self.in_channels = in_channels
        self.width_multiplier = width_multiplier

        for num_blocks, out_channels, kernel_size, stride in blocks:
            out_channels = int(out_channels * width_multiplier)
            blocks = []
            for index in range(num_blocks):
                if index == 0:
                    blocks.append(
                        SparseConvBlock(
                            in_channels,
                            out_channels,
                            kernel_size,
                            stride=stride,
                        ))
                else:
                    blocks.append(
                        SparseResBlock(
                            in_channels,
                            out_channels,
                            kernel_size,
                        ))
                in_channels = out_channels
            self.append(nn.Sequential(*blocks))

    def forward(self, x: SparseTensor) -> List[SparseTensor]:
        outputs = []
        for module in self:
            x = module(x)
            outputs.append(x)
        return outputs


class SparseResNet21D(SparseResNet):

    def __init__(self, **kwargs) -> None:
        super().__init__(
            blocks=[
                (3, 16, 3, 1),
                (3, 32, 3, 2),
                (3, 64, 3, 2),
                (3, 128, 3, 2),
                (1, 128, (1, 3, 1), (1, 2, 1)),
            ],
            **kwargs,
        )

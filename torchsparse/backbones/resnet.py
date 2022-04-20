from typing import Dict, List, Optional, Tuple, Union

import torch.nn as nn

from torchsparse.backbones.modules import SparseConvBlock, SparseResBlock
from torchsparse.tensor import SparseTensor

__all__ = ['SparseResNet', 'sparseresnet18']


class SparseResNet(nn.Module):

    def __init__(
            self,
            layers: List[int],
            in_channels: int = 4,
            channel_sizes: Optional[List[int]] = None,
            width_multiplier: float = 1.0,
            out_kernel_size: Union[int, List[int], Tuple[int, ...]] = None,
            out_stride: Union[int, List[int], Tuple[int, ...]] = None) -> None:

        super().__init__()

        self.in_channels = in_channels
        if out_stride is None:
            out_stride = [1, 2, 1]
        out_ks = [1, 3, 1] if out_kernel_size is None else out_kernel_size
        cs = [16, 32, 64, 128] if channel_sizes is None else channel_sizes
        cs = [int(width_multiplier * x) for x in cs]

        self.stem = self._make_layer(cs[0], layers[0], stride=1)
        self.stage1 = self._make_layer(cs[1], layers[1], stride=2)
        self.stage2 = self._make_layer(cs[2], layers[2], stride=2)
        self.stage3 = self._make_layer(cs[3], layers[3], stride=2)
        self.stage4 = SparseConvBlock(cs[3], cs[3], out_ks, stride=out_stride)

    def _make_layer(
            self, out_channels: int, blocks: int,
            stride: Union[int, List[int], Tuple[int, ...]]) -> nn.Sequential:
        layers = [
            SparseConvBlock(self.in_channels, out_channels, 3, stride=stride)
        ]
        for _ in range(blocks):
            layers.append(SparseResBlock(out_channels, out_channels, 3))
        self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: SparseTensor) -> Dict[str, SparseTensor]:
        feats = {}

        feats['stem'] = self.stem(x)
        feats['stage1'] = self.stage1(feats['stem'])
        feats['stage2'] = self.stage2(feats['stage1'])
        feats['stage3'] = self.stage3(feats['stage2'])
        feats['out'] = self.stage4(feats['stage3'])
        return feats


def sparseresnet18(**kwargs) -> SparseResNet:
    return SparseResNet(layers=[2, 2, 2, 2], **kwargs)

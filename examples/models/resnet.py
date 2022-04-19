import torch.nn as nn
from models.modules.blocks import SparseConvBlock, SparseResBlock

from torchsparse.tensor import SparseTensor

__all__ = ['SparseResNet18']


class SparseResNet18(nn.Module):

    def __init__(self,
                 in_channels: int = 4,
                 channel_sizes: list = None,
                 width_multiplier: float = 1.0) -> None:

        super().__init__()

        cs = channel_sizes if channel_sizes is not None else [16, 32, 64, 128]
        cs = [int(width_multiplier * x) for x in cs]

        self.stem = nn.Sequential(
            SparseConvBlock(in_channels, cs[0], 3, stride=1),
            SparseResBlock(cs[0], cs[0], 3),
            SparseResBlock(cs[0], cs[0], 3),
        )
        self.stage1 = nn.Sequential(
            SparseConvBlock(cs[0], cs[1], 3, stride=2),
            SparseResBlock(cs[1], cs[1], 3),
            SparseResBlock(cs[1], cs[1], 3),
        )
        self.stage2 = nn.Sequential(
            SparseConvBlock(cs[1], cs[2], 3, stride=2),
            SparseResBlock(cs[2], cs[2], 3),
            SparseResBlock(cs[2], cs[2], 3),
        )
        self.stage3 = nn.Sequential(
            SparseConvBlock(cs[2], cs[3], 3, stride=2),
            SparseResBlock(cs[3], cs[3], 3),
            SparseResBlock(cs[3], cs[3], 3),
        )

        self.stage4 = SparseConvBlock(cs[3], cs[3], [1, 3, 1], stride=[1, 2, 1])

    def forward(self, x: SparseTensor):
        feats = {}

        feats['stem'] = self.stem(x)
        feats['stage1'] = self.stage1(feats['stem'])
        feats['stage2'] = self.stage2(feats['stage1'])
        feats['stage3'] = self.stage3(feats['stage2'])
        feats['out'] = self.stage4(feats['stage3'])
        return feats

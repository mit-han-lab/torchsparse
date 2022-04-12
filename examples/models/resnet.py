import torch.nn as nn
from models.modules.layers_3d import SparseConvBlock, SparseResBlock

__all__ = ['SparseResNet']


class SparseResNet(nn.Module):

    def __init__(self, in_channels: int = 4) -> None:
        super().__init__()

        cs = [16, 32, 64, 128]

        self.stem = nn.Sequential(
            SparseConvBlock(in_channels, cs[0], 3, stride=1),
            SparseResBlock(cs[0], cs[0], 3), SparseResBlock(cs[0], cs[0], 3))
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

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feats = {}

        feats['stem'] = self.stem(x)
        feats['stage1'] = self.stage1(feats['stem'])
        feats['stage2'] = self.stage2(feats['stage1'])
        feats['stage3'] = self.stage3(feats['stage2'])
        feats['out'] = self.stage4(feats['stage3'])
        return feats

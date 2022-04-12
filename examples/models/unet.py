import torch.nn as nn
from models.modules.layers_3d import (SparseConvBlock, SparseDeConvBlock,
                                      SparseResBlock)

import torchsparse
import torchsparse.nn as spnn

__all__ = ['SparseResUNet']


class SparseResUNet(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        cr = kwargs.get('cr', 1.0)
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]

        self.stem = nn.Sequential(
            spnn.Conv3d(4, cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True))

        self.stage1 = nn.Sequential(
            SparseConvBlock(cs[0], cs[0], kernel_size=2, stride=2, dilation=1),
            SparseResBlock(cs[0], cs[1], kernel_size=3, stride=1, dilation=1),
            SparseResBlock(cs[1], cs[1], kernel_size=3, stride=1, dilation=1),
        )

        self.stage2 = nn.Sequential(
            SparseConvBlock(cs[1], cs[1], kernel_size=2, stride=2, dilation=1),
            SparseResBlock(cs[1], cs[2], kernel_size=3, stride=1, dilation=1),
            SparseResBlock(cs[2], cs[2], kernel_size=3, stride=1, dilation=1))

        self.stage3 = nn.Sequential(
            SparseConvBlock(cs[2], cs[2], kernel_size=2, stride=2, dilation=1),
            SparseResBlock(cs[2], cs[3], kernel_size=3, stride=1, dilation=1),
            SparseResBlock(cs[3], cs[3], kernel_size=3, stride=1, dilation=1),
        )

        self.stage4 = nn.Sequential(
            SparseConvBlock(cs[3], cs[3], kernel_size=2, stride=2, dilation=1),
            SparseResBlock(cs[3], cs[4], kernel_size=3, stride=1, dilation=1),
            SparseResBlock(cs[4], cs[4], kernel_size=3, stride=1, dilation=1),
        )

        self.up1 = nn.ModuleList([
            SparseDeConvBlock(cs[4], cs[5], kernel_size=2, stride=2),
            nn.Sequential(
                SparseResBlock(cs[5] + cs[3],
                               cs[5],
                               kernel_size=3,
                               stride=1,
                               dilation=1),
                SparseResBlock(cs[5],
                               cs[5],
                               kernel_size=3,
                               stride=1,
                               dilation=1),
            )
        ])

        self.up2 = nn.ModuleList([
            SparseDeConvBlock(cs[5], cs[6], kernel_size=2, stride=2),
            nn.Sequential(
                SparseResBlock(cs[6] + cs[2],
                               cs[6],
                               kernel_size=3,
                               stride=1,
                               dilation=1),
                SparseResBlock(cs[6],
                               cs[6],
                               kernel_size=3,
                               stride=1,
                               dilation=1),
            )
        ])

        self.up3 = nn.ModuleList([
            SparseDeConvBlock(cs[6], cs[7], kernel_size=2, stride=2),
            nn.Sequential(
                SparseResBlock(cs[7] + cs[1],
                               cs[7],
                               kernel_size=3,
                               stride=1,
                               dilation=1),
                SparseResBlock(cs[7],
                               cs[7],
                               kernel_size=3,
                               stride=1,
                               dilation=1),
            )
        ])

        self.up4 = nn.ModuleList([
            SparseDeConvBlock(cs[7], cs[8], kernel_size=2, stride=2),
            nn.Sequential(
                SparseResBlock(cs[8] + cs[0],
                               cs[8],
                               kernel_size=3,
                               stride=1,
                               dilation=1),
                SparseResBlock(cs[8],
                               cs[8],
                               kernel_size=3,
                               stride=1,
                               dilation=1),
            )
        ])

        self.weight_initialization()
        self.dropout = nn.Dropout(0.3, True)

    def weight_initialization(self):
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
        feats['stage4'] = self.stage4(feats['stage3'])

        y1 = self.up1[0](feats['stage4'])
        y1 = torchsparse.cat([y1, feats['stage3']])
        feats['up1'] = self.up1[1](y1)

        y2 = self.up2[0](feats['up1'])
        y2 = torchsparse.cat([y2, feats['stage2']])
        feats['up2'] = self.up2[1](y2)

        y3 = self.up3[0](feats['up2'])
        y3 = torchsparse.cat([y3, feats['stage1']])
        feats['up3'] = self.up3[1](y3)

        y4 = self.up4[0](feats['up3'])
        y4 = torchsparse.cat([y4, feats['stem']])
        feats['out'] = self.up4[1](y4)

        return feats

import torch
from torch import nn

from torchsparse import SparseTensor
from torchsparse.nn.utils import fapply

__all__ = ['BatchNorm', 'GroupNorm']


class BatchNorm(nn.BatchNorm1d):

    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)


class GroupNorm(nn.GroupNorm):

    def forward(self, input: SparseTensor) -> SparseTensor:
        feats = input.F
        coords = input.C
        stride = input.s
        # PyTorch's GroupNorm function expects the input to be in (N, C, *)
        # format where N is batch size, and C is number of channels. "feats"
        # is not in that format. So, we extract the feats corresponding to
        # each sample, bring it to the format expected by PyTorch's GroupNorm
        # function, and invoke it.
        batch_size = coords[-1][-1] + 1
        num_channels = feats.shape[1]
        new_feats = torch.zeros_like(feats)
        for sample_idx in range(batch_size):
            indices = coords[:, -1] == sample_idx
            sample_feats = feats[indices]
            sample_feats = torch.transpose(sample_feats, 0, 1)
            sample_feats = sample_feats.reshape(1, num_channels, -1)
            normalized_feats = super().forward(sample_feats)
            normalized_feats = normalized_feats.reshape(num_channels, -1)
            normalized_feats = torch.transpose(normalized_feats, 0, 1)
            new_feats[indices] = normalized_feats
        output = SparseTensor(coords=coords, feats=new_feats, stride=stride)
        output.cmaps = input.cmaps
        output.kmaps = input.kmaps
        return output

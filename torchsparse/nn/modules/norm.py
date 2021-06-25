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
        coords, feats, stride = input.coords, input.feats, input.stride

        batch_size = torch.max(coords[:, -1]).item() + 1
        num_channels = feats.shape[1]

        # PyTorch's GroupNorm function expects the input to be in (N, C, *)
        # format where N is batch size, and C is number of channels. "feats"
        # is not in that format. So, we extract the feats corresponding to
        # each sample, bring it to the format expected by PyTorch's GroupNorm
        # function, and invoke it.
        nfeats = torch.zeros_like(feats)
        for k in range(batch_size):
            indices = coords[:, -1] == k
            bfeats = feats[indices]
            bfeats = bfeats.transpose(0, 1).reshape(1, num_channels, -1)
            bfeats = super().forward(bfeats)
            bfeats = bfeats.reshape(num_channels, -1).transpose(0, 1)
            nfeats[indices] = bfeats

        output = SparseTensor(coords=coords, feats=nfeats, stride=stride)
        output.cmaps = input.cmaps
        output.kmaps = input.kmaps
        return output

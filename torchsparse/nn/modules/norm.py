import torch
from torch import nn
from torchsparse.sparse_tensor import *

__all__ = ['BatchNorm', 'GroupNorm']


class BatchNorm(nn.BatchNorm1d):
    def __init__(self,
                 num_features: int,
                 *,
                 eps: float = 1e-5,
                 momentum: float = 0.1) -> None:
        super().__init__(num_features=num_features, eps=eps, momentum=momentum)

    def forward(self, inputs):
        feats = inputs.F
        coords = inputs.C
        stride = inputs.s
        feats = super().forward(feats)
        outputs = SparseTensor(coords=coords, feats=feats, stride=stride)
        outputs.coord_maps = inputs.coord_maps
        outputs.kernel_maps = inputs.kernel_maps
        return outputs


class GroupNorm(nn.GroupNorm):
    def __init__(self,
                 num_groups: int,
                 num_channels: int,
                 eps: float = 1e-5,
                 affine: bool = True) -> None:
        super().__init__(num_groups, num_channels, eps=eps, affine=affine)

    def forward(self, inputs):
        feats = inputs.F
        coords = inputs.C
        stride = inputs.s
        # PyTorch's GroupNorm function expects the input to be in (N, C, *) format where
        # N is batch size, and C is number of channels. "feats" is not in that format.
        # So, we extract the feats corresponding to each sample, bring it to the format
        # expected by PyTorch's GroupNorm function, and invoke it. 
        batch_size = coords[-1][-1] + 1  
        num_channels = feats.shape[1]
        new_feats = torch.zeros_like(feats)
        for sample_idx in range(batch_size):
            indices = coords[:,-1] == sample_idx
            sample_feats = feats[indices]
            sample_feats = torch.transpose(sample_feats, 0, 1)
            sample_feats = sample_feats.reshape(1, num_channels, -1) # N=1. since we have a single sample here
            normalized_feats = super().forward(sample_feats)
            normalized_feats = normalized_feats.reshape(num_channels, -1)
            normalized_feats = torch.transpose(normalized_feats, 0, 1)
            new_feats[indices] = normalized_feats
        outputs = SparseTensor(coords=coords, feats=new_feats, stride=stride)
        outputs.coord_maps = inputs.coord_maps
        outputs.kernel_maps = inputs.kernel_maps
        return outputs

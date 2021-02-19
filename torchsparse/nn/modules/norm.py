from torch import nn
from torchsparse.sparse_tensor import *

__all__ = ['BatchNorm', 'LayerNorm']


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


class LayerNorm(nn.LayerNorm):
    def __init__(self,
                 normalized_shape,
                 eps: float = 1e-5,
                 elementwise_affine: bool = True) -> None:
        super().__init__(normalized_shape,
                         eps=eps,
                         elementwise_affine=elementwise_affine)

    def forward(self, inputs):
        feats = inputs.F
        coords = inputs.C
        stride = inputs.s
        feats = super().forward(feats)
        outputs = SparseTensor(coords=coords, feats=feats, stride=stride)
        outputs.coord_maps = inputs.coord_maps
        outputs.kernel_maps = inputs.kernel_maps
        return outputs

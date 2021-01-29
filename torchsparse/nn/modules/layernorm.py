from torch import nn

from torchsparse.sparse_tensor import *

__all__ = ['LayerNorm']


class LayerNorm(nn.LayerNorm):
    def __init__(self,
                 normalized_shape,
                 eps=1e-05,
                 elementwise_affine=True):
        super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, inputs):
        features = inputs.F
        coords = inputs.C
        cur_stride = inputs.s
        output_features = super().forward(features)
        output_tensor = SparseTensor(output_features, coords, cur_stride)
        output_tensor.coord_maps = inputs.coord_maps
        output_tensor.kernel_maps = inputs.kernel_maps
        return output_tensor

from torch import nn

from torchsparse.sparse_tensor import *

__all__ = ['BatchNorm']


class BatchNorm(nn.BatchNorm1d):
    def __init__(self,
                 num_features: int,
                 *,
                 eps: float = 1e-5,
                 momentum: float = 0.1) -> None:
        super().__init__(num_features=num_features, eps=eps, momentum=momentum)

    def forward(self, inputs):
        features = inputs.F
        coords = inputs.C
        cur_stride = inputs.s
        output_features = super().forward(features)
        output_tensor = SparseTensor(output_features, coords, cur_stride)
        output_tensor.coord_maps = inputs.coord_maps
        output_tensor.kernel_maps = inputs.kernel_maps
        return output_tensor

from torch import nn
from torchsparse.sparse_tensor import *

__all__ = ['ReLU', 'LeakyReLU']


class ReLU(nn.ReLU):
    def __init__(self, inplace: bool = True) -> None:
        super().__init__(inplace=inplace)

    def forward(self, inputs):
        coords, feats, stride = inputs.coords, inputs.feats, inputs.stride
        feats = super().forward(feats)
        outputs = SparseTensor(coords, feats, stride=stride)
        outputs.coord_maps = inputs.coord_maps
        outputs.kernel_maps = inputs.kernel_maps
        return outputs


<<<<<<< Updated upstream
class LeakyReLU(nn.LeakyReLU):
    def __init__(self, negative_slope: float = 0.1,
=======

class LeakyReLU(Activation):
    def __init__(self,
                 negative_slope: float = 0.1,
>>>>>>> Stashed changes
                 inplace: bool = True) -> None:
        super().__init__(negative_slope=negative_slope, inplace=inplace)

    def forward(self, inputs):
        coords, feats, stride = inputs.coords, inputs.feats, inputs.stride
        feats = super().forward(feats)
        outputs = SparseTensor(coords, feats, stride=stride)
        outputs.coord_maps = inputs.coord_maps
        outputs.kernel_maps = inputs.kernel_maps
        return outputs

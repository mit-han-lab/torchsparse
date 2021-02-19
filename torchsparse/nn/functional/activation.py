from torch.nn import functional as F
from torchsparse.sparse_tensor import *

__all__ = ['relu', 'leaky_relu']


def relu(inputs: SparseTensor, inplace: bool = True) -> SparseTensor:
    coords, feats, stride = inputs.coords, inputs.feats, inputs.stride
    feats = F.relu(feats, inplace=inplace)
    outputs = SparseTensor(coords, feats, stride=stride)
    outputs.coord_maps = inputs.coord_maps
    outputs.kernel_maps = inputs.kernel_maps
    return outputs


def leaky_relu(inputs: SparseTensor,
               negative_slope: float = 0.1,
               inplace: bool = True) -> SparseTensor:
    coords, feats, stride = inputs.coords, inputs.feats, inputs.stride
    feats = F.leaky_relu(feats, negative_slope=negative_slope, inplace=inplace)
    outputs = SparseTensor(coords, feats, stride=stride)
    outputs.coord_maps = inputs.coord_maps
    outputs.kernel_maps = inputs.kernel_maps
    return outputs

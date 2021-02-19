import functools

from torch.nn import functional as F
from torchsparse.sparse_tensor import *

__all__ = ['spact', 'relu', 'leaky_relu']


def spact(inputs, act_funct=F.relu):
    feats = inputs.feats
    coords = inputs.coords
    stride = inputs.stride
    feats = act_funct(feats)
    outputs = SparseTensor(coords=coords, feats=feats, stride=stride)
    outputs.coord_maps = inputs.coord_maps
    outputs.kernel_maps = inputs.kernel_maps
    return outputs


def relu(inputs, inplace=True):
    return spact(inputs, functools.partial(F.relu, inplace=inplace))


def leaky_relu(inputs, negative_slope=0.1, inplace=True):
    return spact(
        inputs,
        functools.partial(F.leaky_relu,
                          inplace=inplace,
                          negative_slope=negative_slope))

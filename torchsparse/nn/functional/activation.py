import functools

from torch.nn import functional as F
from torchsparse.sparse_tensor import *

__all__ = ['spact', 'sprelu', 'spleaky_relu']


def spact(inputs, act_funct=F.relu):
    feats = inputs.F
    coords = inputs.C
    stride = inputs.s
    output_features = act_funct(feats)
    outputs = SparseTensor(output_features, coords, stride)
    outputs.coord_maps = inputs.coord_maps
    outputs.kernel_maps = inputs.kernel_maps
    return outputs


def sprelu(inputs, inplace=True):
    return spact(inputs, functools.partial(F.relu, inplace=inplace))


def spleaky_relu(inputs, negative_slope=0.1, inplace=True):
    return spact(
        inputs,
        functools.partial(F.leaky_relu,
                          inplace=inplace,
                          negative_slope=negative_slope))

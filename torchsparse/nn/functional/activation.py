import functools

from torch.nn import functional as F

from torchsparse.sparse_tensor import *

__all__ = ['spact', 'sprelu', 'spleaky_relu']


def spact(inputs, act_funct=F.relu):
    features = inputs.F
    coords = inputs.C
    cur_stride = inputs.s
    output_features = act_funct(features)
    output_tensor = SparseTensor(output_features, coords, cur_stride)
    output_tensor.coord_maps = inputs.coord_maps
    output_tensor.kernel_maps = inputs.kernel_maps
    return output_tensor


def sprelu(inputs, inplace=True):
    return spact(inputs, functools.partial(F.relu, inplace=inplace))


def spleaky_relu(inputs, negative_slope=0.1, inplace=True):
    return spact(
        inputs,
        functools.partial(F.leaky_relu,
                          inplace=inplace,
                          negative_slope=negative_slope))

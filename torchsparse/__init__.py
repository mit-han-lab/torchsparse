import torch

from .point_tensor import *
from .sparse_tensor import *

__version__ = '1.2.0'


def cat(input_list, dim=1):
    assert len(input_list) > 0
    inputs = input_list[0]
    features = inputs.F
    coords = inputs.C
    cur_stride = inputs.s
    feats = torch.cat([inputs.F for inputs in input_list], 1)
    output_tensor = SparseTensor(coords, feats, stride=cur_stride)
    output_tensor.coord_maps = inputs.coord_maps
    output_tensor.kernel_maps = inputs.kernel_maps
    return output_tensor

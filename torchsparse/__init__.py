import torch
from .sparse_tensor import *
from .point_tensor import *

__version__ = '1.3.0'


def cat(input_list, dim=1):
    assert len(input_list) > 0
    inputs = input_list[0]
    features = inputs.F
    coords = inputs.C
    cur_stride = inputs.s
    output_tensor = SparseTensor(
        torch.cat([inputs.F for inputs in input_list], 1), coords, cur_stride)
    output_tensor.coord_maps = inputs.coord_maps
    output_tensor.kernel_maps = inputs.kernel_maps
    return output_tensor

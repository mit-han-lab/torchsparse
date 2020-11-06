import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torchsparse.sparse_tensor import SparseTensor

__all__ = ['spcrop']

def spcrop(self, inputs):
    features = inputs.F
    coords = inputs.C
    cur_stride = inputs.s
        
    valid_flag = ((coords[:, :3] >= loc_min) & (coords[:, :3] < loc_max)).all(-1)
    output_coords = coords[valid_flag]
    output_features = features[valid_flag]
    return SparseTensor(output_features, output_coords, cur_stride)
    

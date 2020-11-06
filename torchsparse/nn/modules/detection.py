import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torchsparse.sparse_tensor import SparseTensor
from torchsparse.nn.functional import spcrop

class SparseCrop(nn.Module):
    def __init__(self, loc_min, loc_max):
        super().__init__()
        self.loc_min = torch.cuda.IntTensor([list(loc_min)])
        self.loc_max = torch.cuda.IntTensor([list(loc_max)])
    
    def forward(self, inputs):
        return spcrop(inputs, self.loc_min, self.loc_max)

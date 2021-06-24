import torch
from torch import nn

from torchsparse import SparseTensor
from torchsparse.nn.functional import spcrop

__all__ = ['SparseCrop']


class SparseCrop(nn.Module):

    def __init__(self, loc_min, loc_max):
        super().__init__()
        self.loc_min = torch.cuda.IntTensor([list(loc_min)])
        self.loc_max = torch.cuda.IntTensor([list(loc_max)])

    def forward(self, input: SparseTensor) -> SparseTensor:
        return spcrop(input, self.loc_min, self.loc_max)

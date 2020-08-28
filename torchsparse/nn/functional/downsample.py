import torch
from torch.autograd import Function

import torchsparse_cuda
from torchsparse.nn.functional.hash import *

__all__ = ['spdownsample']


class DownsampleGPU(Function):
    @staticmethod
    def forward(ctx, coords, ratio):
        '''
        Inputs:
        coords: torch.Int32 tensor, N x 4
        ratio: float, downsample ratio
        Outputs:
        coords_downsampled: torch.Int32 tensor, M x 4
        Algorithm: 
        Using torch.unique to get **inverse** indices
        Then use the insertion kernel.
        TBD:
        The insertion kernel w/o atomic op.
        '''
        #coords = coords.to('cpu')
        coords_float = coords[:, :3].float()
        # following Minkowski engine
        coords_new = torch.floor(torch.floor(coords_float / ratio) *
                                 ratio).int()
        coords_new = torch.cat([coords_new, coords[:, 3].view(-1, 1)], 1)
        coords_new_hash = sphash(coords_new)
        uq, inv, cnt = torch.unique(coords_new_hash,
                                    return_inverse=True,
                                    return_counts=True)
        inv = inv.int()
        cnt = cnt.int()
        # rounding is necessary
        # gpu
        if 'cuda' in str(coords.device):
            uq_coords = torch.round(
                torchsparse_cuda.insertion_forward(coords_new.float(), inv,
                                                   cnt))
        elif 'cpu' in str(coords.device):
            uq_coords = torch.round(
                torchsparse_cuda.cpu_insertion_forward(coords_new.float(), inv,
                                                       cnt))
        else:
            device = coords.device
            uq_coords = torch.round(
                torchsparse_cuda.cpu_insertion_forward(
                    coords_new.float().cpu(), inv.cpu(), cnt.cpu()))
            uq_coords = uq_coords.to(device)
        uq_coords = uq_coords.int()

        # Notice: corrds_new_hash cannot be directly used
        return uq_coords  #, coords_new_hash


downsample_gpu = DownsampleGPU.apply


def spdownsample(coords, ratio):
    return downsample_gpu(coords, ratio)

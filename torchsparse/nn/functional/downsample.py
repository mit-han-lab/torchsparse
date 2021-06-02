import torch
import torchsparse_backend
from torch.autograd import Function
from torchsparse.nn.functional.hash import *
from torchsparse.nn.functional.voxelize import spvoxelize

__all__ = ['spdownsample']


class DownsampleGPU(Function):
    @staticmethod
    def forward(ctx, coords, ratio):
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
            uq_coords = torch.round(spvoxelize(coords_new.float(), inv,
                                                      cnt))
        elif 'cpu' in str(coords.device):
            uq_coords = torch.round(
                torchsparse_backend.cpu_insertion_forward(
                    coords_new.float(), inv, cnt))
        else:
            device = coords.device
            uq_coords = torch.round(
                torchsparse_backend.cpu_insertion_forward(
                    coords_new.float().cpu(), inv.cpu(), cnt.cpu()))
            uq_coords = uq_coords.to(device)
        uq_coords = uq_coords.int()

        # Notice: corrds_new_hash cannot be directly used
        return uq_coords  #, coords_new_hash


downsample_gpu = DownsampleGPU.apply


def spdownsample(coords, ratio):
    return downsample_gpu(coords, ratio)

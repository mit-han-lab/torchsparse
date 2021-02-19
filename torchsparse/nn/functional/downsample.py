import torch
import torchsparse_backend
from torchsparse.nn.functional.hash import *

__all__ = ['spdownsample']


def spdownsample(coords, ratio):
    coords_float = coords[:, :3].float()
    coords_new = torch.floor(torch.floor(coords_float / ratio) * ratio).int()
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
            torchsparse_backend.insertion_forward(coords_new.float(), inv,
                                                  cnt))
    elif 'cpu' in str(coords.device):
        uq_coords = torch.round(
            torchsparse_backend.cpu_insertion_forward(coords_new.float(), inv,
                                                      cnt))
    else:
        device = coords.device
        uq_coords = torch.round(
            torchsparse_backend.cpu_insertion_forward(coords_new.float().cpu(),
                                                      inv.cpu(), cnt.cpu()))
        uq_coords = uq_coords.to(device)
    uq_coords = uq_coords.int()

    return uq_coords

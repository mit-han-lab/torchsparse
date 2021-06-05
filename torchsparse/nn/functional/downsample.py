from typing import Union

import torch
import torchsparse_backend
from torch.autograd import Function
from torchsparse.nn.functional.hash import *
from torchsparse.utils.kernel_region import KernelRegion

__all__ = ['spdownsample']



def spdownsample(
    coords: torch.Tensor, 
    ratio: Union[int, torch.Tensor] = 2, 
    kernel_size: Union[int, list, tuple] = 2, 
    tensor_stride: Union[int, torch.Tensor] = 1
):
    if isinstance(kernel_size, int) and isinstance(ratio, int):
        direct_downsample = kernel_size == ratio
    else:
        ratio = ratio.int()
        if isinstance(kernel_size, int):
            # ratio is a permutation of [1, 1, kernel_size]
            direct_downsample = (kernel_size == ratio.prod().item()) & \
                (torch.sum(ratio == kernel_size) == 1).item()
        else:
            direct_downsample = False

    if direct_downsample:
        _ratio = ratio * tensor_stride
        new_coords = torch.cat([
            coords[:, :3] // _ratio * _ratio, coords[:, 3:]
        ], 1)
        return torch.unique(new_coords, dim=0)
    else:
        kernel_region = KernelRegion(kernel_size, tensor_stride, dilation=1)
        # kernel volume x 3
        kernel_offset = kernel_region.get_kernel_offset().to(coords.device)
        new_coords = coords[:, :3].unsqueeze(1).repeat(1, kernel_offset.size(0), 1) + kernel_offset
        # (N x kernel volume) x 4
        new_coords = torch.cat([
            coords[:, 3:].repeat(1, kernel_offset.size(0)).view(-1, 1),
            new_coords.view(-1, 3)
        ], dim=1)
        new_ts = tensor_stride * ratio
        # only keep these coordinates that is multiple of new_ts.
        if isinstance(new_ts, torch.Tensor):
            new_ts = new_ts[0]
            new_coords = new_coords[
                (new_coords[:, 1] % new_ts[0].item() == 0) & (new_coords[:, 2] % new_ts[1].item() == 0) & \
                (new_coords[:, 3] % new_ts[2].item() == 0)
            ]
        else:
            new_coords = new_coords[
                (new_coords[:, 1] % new_ts == 0) & (new_coords[:, 2] % new_ts == 0) & \
                (new_coords[:, 3] % new_ts == 0)
            ]
        new_coords = new_coords[
            (new_coords[:, 1] >= 0) & (new_coords[:, 2] >= 0) & (new_coords[:, 3] >= 0) 
        ]
        # filter out duplicates
        new_coords = torch.unique(new_coords, dim=0)
        new_coords = new_coords[:, [1, 2, 3, 0]]
        return new_coords

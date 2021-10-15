from typing import Tuple, Union

import torch

from torchsparse.nn.utils import get_kernel_offsets
from torchsparse.utils import make_ntuple

__all__ = ['spdownsample']


def spdownsample(
        coords: torch.Tensor,
        stride: Union[int, Tuple[int, ...]] = 2,
        kernel_size: Union[int, Tuple[int, ...]] = 2,
        tensor_stride: Union[int, Tuple[int, ...]] = 1) -> torch.Tensor:
    stride = make_ntuple(stride, ndim=3)
    kernel_size = make_ntuple(kernel_size, ndim=3)
    tensor_stride = make_ntuple(tensor_stride, ndim=3)

    sample_stride = [stride[k] * tensor_stride[k] for k in range(3)]
    sample_stride = torch.tensor(sample_stride,
                                 dtype=torch.int,
                                 device=coords.device).unsqueeze(dim=0)

    if all(stride[k] in [1, kernel_size[k]] for k in range(3)):
        coords = coords.clone()
        coords[:, :3] = torch.div(coords[:, :3],
                                  sample_stride).trunc() * sample_stride
    else:
        offsets = get_kernel_offsets(kernel_size,
                                     tensor_stride,
                                     device=coords.device)
        kernel_volume = offsets.size(0)

        coords_min = torch.min(coords[:, :3], dim=0, keepdim=True).values

        x = coords[:, :3].unsqueeze(dim=1).repeat(1, kernel_volume, 1) + offsets
        b = coords[:, 3:].repeat(1, kernel_volume)
        coords = torch.cat([x.view(-1, 3), b.view(-1, 1)], dim=1)

        # TODO(Zhijian): We need to also filter `coords` based on `coords_max`.
        mask = (coords[:, :3] % sample_stride == 0)
        mask &= (coords[:, :3] >= coords_min)
        mask = torch.all(mask, dim=1)
        coords = coords[mask]

    # This makes sure that the points will be ordered with respect to the batch
    # index, but this will not affect the correctness of the result.
    coords = coords[:, [3, 0, 1, 2]]
    coords = torch.unique(coords, dim=0)
    coords = coords[:, [1, 2, 3, 0]]
    return coords

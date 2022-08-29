from typing import Tuple, Union

import torch

import torchsparse.backend
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
        coords[:, :3] = torch.div(
            coords[:, :3],
            sample_stride.float()).trunc() * sample_stride  # type: ignore
        coords = coords[:, [3, 0, 1, 2]]
        coords = torch.unique(coords, dim=0)
        coords = coords[:, [1, 2, 3, 0]]
        return coords
    else:
        if coords.device.type == 'cuda':
            coords = coords[:, [3, 0, 1, 2]]
            out_coords = torchsparse.backend.downsample_cuda(
                coords,
                coords.max(0).values,
                coords.min(0).values, kernel_size, stride,
                tensor_stride)[:, [1, 2, 3, 0]]
            return out_coords
        else:
            raise NotImplementedError

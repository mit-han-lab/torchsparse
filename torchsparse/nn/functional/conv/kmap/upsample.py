from typing import Tuple, Union, Optional

import torch

import torchsparse.backend
from torchsparse.utils import make_ntuple, make_tensor
from torchsparse.nn.utils.kernel import get_kernel_offsets

__all__ = ["spupsample_generative"]


def spupsample_generative(
    _coords: torch.Tensor,
    stride: Union[int, Tuple[int, ...]] = 2,
    kernel_size: Union[int, Tuple[int, ...]] = 2,
    padding: torch.Tensor = 0,
    spatial_range: Optional[Tuple[int]] = None,
) -> torch.Tensor:
    stride = make_ntuple(stride, ndim=3)
    kernel_size = make_ntuple(kernel_size, ndim=3)
    padding = make_ntuple(padding, ndim=3)
    sample_stride = make_tensor(
        stride, dtype=torch.int, device=_coords.device
    ).unsqueeze(0)
    # stride and dilation are both 1
    kernel_offsets = get_kernel_offsets(kernel_size, 1, 1, device=_coords.device)
    coords = _coords.clone()
    coords[:, 1:] *= sample_stride
    coords = coords.unsqueeze(1).repeat(1, kernel_offsets.size(0), 1)
    coords[:, :, 1:] = coords[:, :, 1:] + kernel_offsets.unsqueeze(0)
    assert (
        spatial_range is not None
    ), "spatial range must be specified in generative mode"
    for i in range(1, coords.size(-1)):
        coords[:, :, i].clamp_(min=0, max=spatial_range[i] - 1)
    coords = coords.reshape(-1, coords.size(-1))
    coords = torch.unique(coords, dim=0)
    return coords

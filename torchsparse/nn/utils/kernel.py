from typing import Tuple, Union

import numpy as np
import torch

from torchsparse.utils import make_ntuple

__all__ = ['get_kernel_offsets']


def get_kernel_offsets(size: Union[int, Tuple[int, ...]],
                       stride: Union[int, Tuple[int, ...]] = 1,
                       dilation: Union[int, Tuple[int, ...]] = 1,
                       device: str = 'cpu') -> torch.Tensor:
    size = make_ntuple(size, ndim=3)
    stride = make_ntuple(stride, ndim=3)
    dilation = make_ntuple(dilation, ndim=3)

    offsets = [(np.arange(-size[k] // 2 + 1, size[k] // 2 + 1) * stride[k]
                * dilation[k]) for k in range(3)]

    # This condition check is only to make sure that our weight layout is
    # compatible with `MinkowskiEngine`.
    if np.prod(size) % 2 == 1:
        offsets = [[x, y, z] for z in offsets[2] for y in offsets[1]
                   for x in offsets[0]]
    else:
        offsets = [[x, y, z] for x in offsets[0] for y in offsets[1]
                   for z in offsets[2]]

    offsets = torch.tensor(offsets, dtype=torch.int, device=device)
    return offsets

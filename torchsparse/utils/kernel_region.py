from typing import Tuple, Union

import numpy as np
import torch

__all__ = ['KernelRegion']


class KernelRegion:
    def __init__(self,
                 size: Union[int, Tuple[int, int, int]],
                 stride: int = 1,
                 dilation: int = 1) -> None:
        self.kernel_size = size
        self.stride = stride
        self.dilation = dilation

        if isinstance(size, int):
            size = [size] * 3

        offsets = [
            np.arange(-size[k] // 2 + 1, size[k] // 2 + 1) * stride * dilation
            for k in range(3)
        ]
        offsets = [[x, y, z] for x in offsets[0] for y in offsets[1]
                   for z in offsets[1]]
        self.offsets = torch.tensor(offsets, dtype=torch.int)

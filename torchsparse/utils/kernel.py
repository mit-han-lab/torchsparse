from collections import namedtuple
import numpy as np
import torch
from typing import Union, List, Tuple

__all__ = ['KernelRegion', 'KernelMapKey']

KernelMapKey = namedtuple(
    'KernelMapKey', ['kernel_size', 'cur_stride', 'stride', 'dilation'])

class KernelRegion:
    def __init__(self,
                 kernel_size: Union[int, List[int], Tuple[int, int, int]] = 3,
                 tensor_stride: Union[int, List[int], Tuple[int, int, int],
                                      torch.Tensor] = 1,
                 dilation: Union[int, List[int], Tuple[int, int, int]] = 1,
                 dim: List[int] = [0, 1, 2]) -> None:
        self.kernel_size = kernel_size
        if isinstance(tensor_stride, int):
            self.tensor_stride = [tensor_stride] * 3
        elif isinstance(tensor_stride, torch.Tensor):
            self.tensor_stride = tensor_stride.cpu().view(-1).numpy().tolist()
        else:
            self.tensor_stride = tensor_stride
        if isinstance(dilation, int):
            dilation = [dilation] * 3
        self.dilation = dilation
        assert len(self.tensor_stride) == 3, 'Wrong tensor_stride'
        assert len(self.dilation) == 3, 'Wrong dilation'

        ts = self.tensor_stride
        d = self.dilation

        if not isinstance(kernel_size, (list, tuple)):
            if kernel_size % 2 == 0:
                # even
                region_type = 0
            else:
                # odd
                region_type = 1

            self.region_type = region_type

            x_offset = (
                np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1) *
                ts[0] * d[0]).tolist() if 0 in dim else [0]
            y_offset = (
                np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1) *
                ts[1] * d[1]).tolist() if 1 in dim else [0]
            z_offset = (
                np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1) *
                ts[2] * d[2]).tolist() if 2 in dim else [0]

            if self.region_type == 1:
                kernel_offset = [[x, y, z] for z in z_offset for y in y_offset
                                 for x in x_offset]
            else:
                kernel_offset = [[x, y, z] for x in x_offset for y in y_offset
                                 for z in z_offset]
            kernel_offset = np.array(kernel_offset)
            self.kernel_offset = torch.from_numpy(kernel_offset).int()
        else:
            if dim == [0, 1, 2] and len(kernel_size) == 3:
                kernel_x_size = kernel_size[0]
                kernel_y_size = kernel_size[1]
                kernel_z_size = kernel_size[2]

                x_offset = (np.arange(-kernel_x_size // 2 + 1,
                                      kernel_x_size // 2 + 1) * ts[0] *
                            d[0]).tolist()
                y_offset = (np.arange(-kernel_y_size // 2 + 1,
                                      kernel_y_size // 2 + 1) * ts[1] *
                            d[1]).tolist()
                z_offset = (np.arange(-kernel_z_size // 2 + 1,
                                      kernel_z_size // 2 + 1) * ts[2] *
                            d[2]).tolist()

                kernel_offset = [[x, y, z] for x in x_offset for y in y_offset
                                 for z in z_offset]

                kernel_offset = np.array(kernel_offset)
                self.kernel_offset = torch.from_numpy(kernel_offset).int()
            else:
                raise NotImplementedError

    def get_kernel_offset(self):
        return self.kernel_offset

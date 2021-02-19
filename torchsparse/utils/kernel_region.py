import numpy as np
import torch

__all__ = ['KernelRegion']


class KernelRegion:
    def __init__(self,
                 kernel_size: int = 3,
                 tensor_stride: int = 1,
                 dilation: int = 1,
                 dim=[0, 1, 2]) -> None:
        self.kernel_size = kernel_size
        self.tensor_stride = tensor_stride
        self.dilation = dilation

        if not isinstance(kernel_size, (list, tuple)):
            if kernel_size % 2 == 0:
                # even
                region_type = 0
            else:
                # odd
                region_type = 1

            self.region_type = region_type

            single_offset = (
                np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1) *
                tensor_stride * dilation).tolist()

            x_offset = single_offset if 0 in dim else [0]
            y_offset = single_offset if 1 in dim else [0]
            z_offset = single_offset if 2 in dim else [0]

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
                                      kernel_x_size // 2 + 1) * tensor_stride *
                            dilation).tolist()
                y_offset = (np.arange(-kernel_y_size // 2 + 1,
                                      kernel_y_size // 2 + 1) * tensor_stride *
                            dilation).tolist()
                z_offset = (np.arange(-kernel_z_size // 2 + 1,
                                      kernel_z_size // 2 + 1) * tensor_stride *
                            dilation).tolist()

                kernel_offset = [[x, y, z] for x in x_offset for y in y_offset
                                 for z in z_offset]

                kernel_offset = np.array(kernel_offset)
                self.kernel_offset = torch.from_numpy(kernel_offset).int()
            else:
                raise NotImplementedError

    def get_kernel_offset(self):
        return self.kernel_offset

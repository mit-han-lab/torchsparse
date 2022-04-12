import torch

from torchsparse import nn as spnn

__all__ = ['reorder_kernel']


def reorder_kernel(net):
    for child_name, child in net.named_children():
        if isinstance(child, spnn.Conv3d):
            if (len(child.kernel.data.shape) == 3):
                kernel_volume = child.kernel.data.size(0)
                kernel_cont = torch.zeros_like(child.kernel.data)
                ind = 0
                while ind < kernel_volume - 1:
                    kernel_cont[ind] = child.kernel.data[ind // 2]
                    kernel_cont[ind + 1] = \
                        child.kernel.data[kernel_volume - 1 - ind // 2]
                    ind += 2
                if kernel_volume % 2 == 1:
                    kernel_cont[kernel_volume - 1] = \
                        child.kernel.data[kernel_volume // 2]

                child.kernel.data = kernel_cont
                setattr(net, child_name, child)
        else:
            reorder_kernel(child)

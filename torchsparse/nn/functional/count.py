from torch.autograd import Function

import torchsparse_cuda

__all__ = ['spcount']


class CountGPU(Function):
    @staticmethod
    def forward(ctx, idx, num):
        if 'cuda' in str(idx.device):
            outs = torchsparse_cuda.count_forward(idx.contiguous(), num)
        else:
            outs = torchsparse_cuda.cpu_count_forward(idx.contiguous(), num)
        return outs


count_gpu = CountGPU.apply


def spcount(idx, num):
    return count_gpu(idx, num)

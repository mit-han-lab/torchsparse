import torchsparse_backend
from torch.autograd import Function

__all__ = ['spcount']


class CountGPU(Function):
    @staticmethod
    def forward(ctx, idx, num):
        if 'cuda' in str(idx.device):
            outs = torchsparse_backend.count_forward(idx.contiguous(), num)
        else:
            outs = torchsparse_backend.cpu_count_forward(idx.contiguous(), num)
        return outs


def spcount(idx, num):
    return CountGPU.apply(idx, num)

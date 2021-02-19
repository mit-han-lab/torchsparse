import torchsparse_cuda
from torch.autograd import Function

__all__ = ['sphash']


class HashGPU(Function):
    @staticmethod
    def forward(ctx, idx):
        if 'cuda' in str(idx.device):
            return torchsparse_cuda.hash_forward(idx.contiguous())
        elif 'cpu' in str(idx.device):
            return torchsparse_cuda.cpu_hash_forward(idx.int().contiguous())
        else:
            device = idx.device
            return torchsparse_cuda.cpu_hash_forward(
                idx.int().contiguous().cpu()).to(device)


class KernelHashGPU(Function):
    @staticmethod
    def forward(ctx, idx, koffset):
        if 'cuda' in str(idx.device):
            return torchsparse_cuda.kernel_hash_forward(
                idx.contiguous(), koffset.contiguous())
        elif 'cpu' in str(idx.device):
            return torchsparse_cuda.cpu_kernel_hash_forward(
                idx.int().contiguous(),
                koffset.int().contiguous())
        else:
            device = idx.device
            return torchsparse_cuda.cpu_kernel_hash_forward(
                idx.int().contiguous().cpu(),
                koffset.int().contiguous().cpu()).to(device)


def sphash(idx, koffset=None):
    if koffset is None:
        return HashGPU.apply(idx)
    else:
        return KernelHashGPU.apply(idx, koffset)

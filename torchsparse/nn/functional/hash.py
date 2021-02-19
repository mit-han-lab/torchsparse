import torch
import torchsparse_backend
from torch.autograd import Function

__all__ = ['sphash', 'sphashquery']


class HashGPU(Function):
    @staticmethod
    def forward(ctx, idx):
        if 'cuda' in str(idx.device):
            return torchsparse_backend.hash_forward(idx.contiguous())
        elif 'cpu' in str(idx.device):
            return torchsparse_backend.cpu_hash_forward(idx.int().contiguous())
        else:
            device = idx.device
            return torchsparse_backend.cpu_hash_forward(
                idx.int().contiguous().cpu()).to(device)


class KernelHashGPU(Function):
    @staticmethod
    def forward(ctx, idx, koffset):
        if 'cuda' in str(idx.device):
            return torchsparse_backend.kernel_hash_forward(
                idx.contiguous(), koffset.contiguous())
        elif 'cpu' in str(idx.device):
            return torchsparse_backend.cpu_kernel_hash_forward(
                idx.int().contiguous(),
                koffset.int().contiguous())
        else:
            device = idx.device
            return torchsparse_backend.cpu_kernel_hash_forward(
                idx.int().contiguous().cpu(),
                koffset.int().contiguous().cpu()).to(device)


def sphash(idx, koffset=None):
    if koffset is None:
        return HashGPU.apply(idx)
    else:
        return KernelHashGPU.apply(idx, koffset)


class SparseQuery(Function):
    @staticmethod
    def forward(ctx, hash_query, hash_target):
        if len(hash_query.size()) == 2:
            C = hash_query.size(1)
        else:
            C = 1

        idx_target = torch.arange(len(hash_target),
                                  device=hash_query.device,
                                  dtype=torch.long)

        if 'cuda' in str(hash_query.device):
            out, key_buf, val_buf, key = torchsparse_backend.query_forward(
                hash_query.view(-1).contiguous(), hash_target.contiguous(),
                idx_target)
        elif 'cpu' in str(hash_query.device):
            out = torchsparse_backend.cpu_query_forward(
                hash_query.view(-1).contiguous(), hash_target.contiguous(),
                idx_target)
        else:
            device = hash_query.device
            out = torchsparse_backend.cpu_query_forward(
                hash_query.view(-1).contiguous().cpu(),
                hash_target.contiguous().cpu(), idx_target.cpu()).to(device)

        if C > 1:
            out = out.view(-1, C)
        return (out - 1)


def sphashquery(hash_query, hash_target):
    return SparseQuery.apply(hash_query, hash_target)

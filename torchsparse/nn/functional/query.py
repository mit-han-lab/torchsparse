import torch
from torch.autograd import Function

import torchsparse_cuda

__all__ = ['sphashquery']


class SparseQuery(Function):
    @staticmethod
    def forward(ctx, hash_query, hash_target):
        if len(hash_query.size()) == 2:
            N, C = hash_query.size()
        else:
            N = hash_query.size(0)
            C = 1

        idx_target = torch.arange(len(hash_target),
                                  device=hash_query.device,
                                  dtype=torch.long)

        if 'cuda' in str(hash_query.device):
            out, key_buf, val_buf, key = torchsparse_cuda.query_forward(
                hash_query.view(-1).contiguous(), hash_target.contiguous(),
                idx_target)
        elif 'cpu' in str(hash_query.device):
            out = torchsparse_cuda.cpu_query_forward(
                hash_query.view(-1).contiguous(), hash_target.contiguous(),
                idx_target)
        else:
            device = hash_query.device
            out = torchsparse_cuda.cpu_query_forward(
                hash_query.view(-1).contiguous().cpu(),
                hash_target.contiguous().cpu(), idx_target.cpu()).to(device)

        if C > 1:
            out = out.view(-1, C)
        return (out - 1)


sparse_hash_query = SparseQuery.apply


def sphashquery(hash_query, hash_target):
    return sparse_hash_query(hash_query, hash_target)

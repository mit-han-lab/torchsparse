from typing import Optional

import torch
import torchsparse_backend

__all__ = ['sphash', 'sphashquery']


def sphash(coords: torch.Tensor, offsets: Optional[torch.Tensor] = None):
    if offsets is None:
        if coords.device.type == 'cuda':
            return torchsparse_backend.hash_forward(coords.contiguous())
        elif coords.device.type == 'cpu':
            return torchsparse_backend.cpu_hash_forward(
                coords.int().contiguous())
        elif coords.device.type == 'tpu':
            device = coords.device
            return torchsparse_backend.cpu_hash_forward(
                coords.int().contiguous().cpu()).to(device)
    else:
        if coords.device.type == 'cuda':
            return torchsparse_backend.kernel_hash_forward(
                coords.contiguous(), offsets.contiguous())
        elif coords.device.type == 'cpu':
            return torchsparse_backend.cpu_kernel_hash_forward(
                coords.int().contiguous(),
                offsets.int().contiguous())
        elif coords.device.type == 'tpu':
            device = coords.device
            return torchsparse_backend.cpu_kernel_hash_forward(
                coords.int().contiguous().cpu(),
                offsets.int().contiguous().cpu()).to(device)
        else:
            raise NotImplementedError


def sphashquery(hash_query, hash_target):
    if len(hash_query.size()) == 2:
        C = hash_query.size(1)
    else:
        C = 1

    idx_target = torch.arange(len(hash_target),
                              device=hash_query.device,
                              dtype=torch.long)

    if hash_query.device.type == 'cuda':
        out, key_buf, val_buf, key = torchsparse_backend.query_forward(
            hash_query.view(-1).contiguous(), hash_target.contiguous(),
            idx_target)
    elif hash_query.device.type == 'cpu':
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

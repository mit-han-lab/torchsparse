from typing import Optional

import torch

import torchsparse.backend

__all__ = ['sphash']


def sphash(coords: torch.Tensor,
           offsets: Optional[torch.Tensor] = None) -> torch.Tensor:
    assert coords.dtype == torch.int, coords.dtype
    assert coords.ndim == 2 and coords.shape[1] == 4, coords.shape
    coords = coords.contiguous()

    # TODO(Zhijian): We might be able to merge `hash_kernel` and `hash`.
    if offsets is None:
        if coords.device.type == 'cuda':
            return torchsparse.backend.hash_cuda(coords)
        elif coords.device.type == 'cpu':
            return torchsparse.backend.hash_cpu(coords)
        else:
            device = coords.device
            return torchsparse.backend.hash_cpu(coords.cpu()).to(device)
    else:
        assert offsets.dtype == torch.int, offsets.dtype
        assert offsets.ndim == 2 and offsets.shape[1] == 3, offsets.shape
        offsets = offsets.contiguous()

        if coords.device.type == 'cuda':
            return torchsparse.backend.kernel_hash_cuda(coords, offsets)
        elif coords.device.type == 'cpu':
            return torchsparse.backend.kernel_hash_cpu(coords, offsets)
        else:
            device = coords.device
            return torchsparse.backend.kernel_hash_cpu(coords.cpu(),
                                                       offsets.cpu()).to(device)

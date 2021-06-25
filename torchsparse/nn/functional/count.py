import torch

import torchsparse.backend

__all__ = ['spcount']


def spcount(coords: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
    coords = coords.contiguous()
    if coords.device.type == 'cuda':
        return torchsparse.backend.count_cuda(coords, num)
    elif coords.device.type == 'cpu':
        return torchsparse.backend.count_cpu(coords, num)
    else:
        device = coords.device
        return torchsparse.backend.count_cpu(coords.cpu(), num).to(device)

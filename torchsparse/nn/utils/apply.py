from typing import Callable

import torch

from torchsparse import SparseTensor

__all__ = ['fapply']


def fapply(input: SparseTensor, fn: Callable[..., torch.Tensor], *args,
           **kwargs) -> SparseTensor:
    feats = fn(input.feats, *args, **kwargs)
    output = SparseTensor(coords=input.coords, feats=feats, stride=input.stride)
    output.cmaps = input.cmaps
    output.kmaps = input.kmaps
    return output

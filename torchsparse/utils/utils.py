from itertools import repeat
from typing import List, Tuple, Union

import torch

__all__ = ['make_ntuple']


def make_ntuple(x: Union[int, List[int], Tuple[int, ...], torch.Tensor],
                ndim: int) -> Tuple[int, ...]:
    if isinstance(x, int):
        x = tuple(repeat(x, ndim))
    elif isinstance(x, list):
        x = tuple(x)
    elif isinstance(x, torch.Tensor):
        x = tuple(x.view(-1).cpu().numpy().tolist())

    assert isinstance(x, tuple) and len(x) == ndim, x
    return x

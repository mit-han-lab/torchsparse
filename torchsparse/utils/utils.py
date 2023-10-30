from itertools import repeat
from typing import List, Tuple, Union
from functools import lru_cache
import torch

__all__ = ["make_ntuple", "make_tensor", "make_divisible"]


def make_ntuple(
    x: Union[int, List[int], Tuple[int, ...], torch.Tensor], ndim: int
) -> Tuple[int, ...]:
    if isinstance(x, int):
        x = tuple(repeat(x, ndim))
    elif isinstance(x, list):
        x = tuple(x)
    elif isinstance(x, torch.Tensor):
        x = tuple(x.view(-1).cpu().numpy().tolist())

    assert isinstance(x, tuple) and len(x) == ndim, x
    return x


@lru_cache()
def make_tensor(x: Tuple[int, ...], dtype: torch.dtype, device) -> torch.Tensor:
    return torch.tensor(x, dtype=dtype, device=device)


def make_divisible(x: int, divisor: int):
    return (x + divisor - 1) // divisor * divisor

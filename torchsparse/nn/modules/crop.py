from typing import Optional, Tuple

from torch import nn

from torchsparse import SparseTensor
from torchsparse.nn import functional as F

__all__ = ["SparseCrop"]


class SparseCrop(nn.Module):
    def __init__(
        self,
        coords_min: Optional[Tuple[int, ...]] = None,
        coords_max: Optional[Tuple[int, ...]] = None,
    ) -> None:
        super().__init__()
        self.coords_min = coords_min
        self.coords_max = coords_max

    def forward(self, input: SparseTensor) -> SparseTensor:
        return F.spcrop(input, self.coords_min, self.coords_max)

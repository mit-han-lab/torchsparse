from typing import List, Optional

import torch

from torchsparse.tensor import SparseTensor

# from torch_scatter import scatter_sum

__all__ = ["cat", "generative_add"]


def cat(inputs: List[SparseTensor]) -> SparseTensor:
    feats = torch.cat([input.feats for input in inputs], dim=1)
    output = SparseTensor(coords=inputs[0].coords, feats=feats, stride=inputs[0].stride)
    output._caches = inputs[0]._caches
    return output


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


def generative_add(a: SparseTensor, b: SparseTensor) -> SparseTensor:
    input_a = a if a.F.size(0) >= b.F.size(0) else b
    input_b = b if a.F.size(0) >= b.F.size(0) else a
    union_coords = torch.cat([input_a.C, input_b.C], dim=0)
    union_features = torch.cat([input_a.F, input_b.F], dim=0)
    unique_coords, unique_idx = torch.unique(union_coords, dim=0, return_inverse=True)
    out_feature = scatter_sum(union_features, unique_idx, dim=0)
    out_tensor = SparseTensor(
        out_feature, unique_coords, input_a.s, spatial_range=input_a.spatial_range
    )
    out_tensor._caches = input_a._caches
    return out_tensor

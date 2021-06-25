from typing import Optional, Tuple

import torch

from torchsparse import SparseTensor

__all__ = ['spcrop']


def spcrop(input: SparseTensor,
           coords_min: Optional[Tuple[int, ...]] = None,
           coords_max: Optional[Tuple[int, ...]] = None) -> SparseTensor:
    coords, feats, stride = input.coords, input.feats, input.stride

    mask = torch.ones((coords.shape[0], 3),
                      dtype=torch.bool,
                      device=coords.device)
    if coords_min is not None:
        coords_min = torch.tensor(coords_min,
                                  dtype=torch.int,
                                  device=coords.device).unsqueeze(dim=0)
        mask &= (coords[:, :3] >= coords_min)
    if coords_max is not None:
        coords_max = torch.tensor(coords_max,
                                  dtype=torch.int,
                                  device=coords.device).unsqueeze(dim=0)
        # Using "<" instead of "<=" is for the backward compatability (in
        # some existing detection codebase). We might need to reflect this
        # in the document or change it back to "<=" in the future.
        mask &= (coords[:, :3] < coords_max)

    mask = torch.all(mask, dim=1)
    coords, feats = coords[mask], feats[mask]
    output = SparseTensor(coords=coords, feats=feats, stride=stride)
    return output

from itertools import repeat
from typing import Tuple, Union

import numpy as np

__all__ = ['sparse_quantize']


def ravel_hash(x: np.ndarray) -> np.ndarray:
    assert x.ndim == 2, x.shape

    x -= x.min(axis=0)
    x = x.astype(np.uint64, copy=False)
    xmax = x.max(axis=0).astype(np.uint64) + 1

    h = np.zeros(x.shape[0], dtype=np.uint64)
    for j in range(x.shape[1] - 1):
        h += x[:, j]
        h *= xmax[j + 1]
    h += x[:, -1]
    return h


def sparse_quantize(
        coords,
        voxel_size: Union[float, Tuple[float, ...]] = 1,
        *,
        return_index: bool = False,
        return_inverse: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    if isinstance(voxel_size, (float, int)):
        voxel_size = tuple(repeat(voxel_size, 3))
    assert isinstance(voxel_size, tuple) and len(voxel_size) == 3

    voxel_size = np.array(voxel_size)
    coords = np.floor(coords / voxel_size).astype(np.int32)

    _, indices, inverse_indices = np.unique(ravel_hash(coords),
                                            return_index=True,
                                            return_inverse=True)
    coords = coords[indices]

    output = [coords]
    if return_index:
        output += [indices]
    if return_inverse:
        output += [inverse_indices]
    return output[0] if len(output) == 1 else tuple(output)

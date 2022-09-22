from itertools import repeat
from typing import List, Tuple, Union

import numpy as np

__all__ = ['sparse_quantize']


def ravel_hash(x: np.ndarray) -> np.ndarray:
    assert x.ndim == 2, x.shape

    x = x - np.min(x, axis=0)
    x = x.astype(np.uint64, copy=False)
    xmax = np.max(x, axis=0).astype(np.uint64) + 1

    h = np.zeros(x.shape[0], dtype=np.uint64)
    for k in range(x.shape[1] - 1):
        h += x[:, k]
        h *= xmax[k + 1]
    h += x[:, -1]
    return h


def sparse_quantize(coords,
                    voxel_size: Union[float, Tuple[float, ...]] = 1,
                    *,
                    return_index: bool = False,
                    return_inverse: bool = False) -> List[np.ndarray]:
    """Voxelize x, y, z coordinates and remove duplicates.
    
    Args:
        coords (np.ndarray): An Nx3 array of x, y, z coordinates.
        voxel_size (Union[float, Tuple[float, ...]]): The voxel size.
        return_index (bool): Whether to return the indices of the voxels.
        return_inverse (bool): Whether to return the indices of the original.

    Returns:
        List[np.ndarray]: A list of voxelized coordinates.
    """
    if isinstance(voxel_size, (float, int)):
        voxel_size = tuple(repeat(voxel_size, 3))
    assert isinstance(voxel_size, tuple) and len(voxel_size) == 3

    voxel_size = np.array(voxel_size)
    coords = np.floor(coords / voxel_size).astype(np.int32)

    _, indices, inverse_indices = np.unique(ravel_hash(coords),
                                            return_index=True,
                                            return_inverse=True)
    coords = coords[indices]

    outputs = [coords]
    if return_index:
        outputs += [indices]
    if return_inverse:
        outputs += [inverse_indices]
    return outputs[0] if len(outputs) == 1 else outputs

from typing import Any, Dict, Tuple, Union, Optional, List

import numpy as np
import torch
from torch import nn

import torchsparse
from torchsparse import nn as spnn
from torchsparse.utils import make_ntuple, to_dense

from .test_utils import generate_feature_map

__all__ = ["test_to_dense_forward"]


def test_to_dense_forward(
    batch_size: int = 1,
    shape: Union[int, Tuple[int, ...]] = 3,
    num_points: int = 6,
    channel: int = 4,
    device="cuda:0",
):

    np.random.seed(0)
    torch.manual_seed(0)

    torch_dtype = torch.float16
    np_dtype = np.float16

    shape = make_ntuple(shape, ndim=3)
    spatial_range = make_ntuple([batch_size, *shape], ndim=4)

    if num_points > np.prod(shape):
        print("Warning: num_points exceeds coords range!")
        print("         reduce num_points to %d!" % np.prod(shape))
        num_points = np.prod(shape)
    num_points = [num_points] * batch_size

    sparse_dict = generate_feature_map(shape, num_points, channel, dtype=np_dtype)

    feats = np.ascontiguousarray(sparse_dict["feats"])
    coords = np.ascontiguousarray(sparse_dict["coords"][:, [3, 0, 1, 2]])  # batch first
    ref_dense_feats = sparse_dict["dense_feats"].transpose(0, 2, 3, 4, 1)

    coords_t = torch.from_numpy(coords).int().to(device)
    feats_t = torch.from_numpy(feats).to(torch_dtype).to(device)

    output = to_dense(feats_t, coords_t, spatial_range).cpu().numpy()

    # print(output)
    # print(ref_dense_feats)

    max_adiff = np.max(np.abs(output - ref_dense_feats))
    return max_adiff


if __name__ == "__main__":
    max_adiff = test_to_dense_forward()
    print(max_adiff)

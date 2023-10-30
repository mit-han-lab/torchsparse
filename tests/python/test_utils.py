from typing import Any, Dict, Tuple, Union, Optional, List

import numpy as np
import torch


def generate_feature_map(
    shape,
    num_points,
    num_channels,
    data_range=(-1, 1),
    with_dense=True,
    dtype=np.float16,
):
    dense_shape = shape
    ndim = len(dense_shape)
    num_points = np.array(num_points)
    batch_size = len(num_points)
    batch_indices = []
    coords_total = np.stack(np.meshgrid(*[np.arange(0, s) for s in shape]), axis=-1)
    coords_total = coords_total.reshape(-1, ndim)

    for i in range(batch_size):
        np.random.shuffle(coords_total)
        inds_total = coords_total[: num_points[i]]
        inds_total = np.pad(
            inds_total,
            ((0, 0), (0, 1)),  # batch last
            mode="constant",
            constant_values=i,
        )
        batch_indices.append(inds_total)

    features = np.random.uniform(
        data_range[0], data_range[1], size=[num_points.sum(), num_channels]
    ).astype(dtype)

    sparse_dict = dict(
        [
            ("feats", features),
        ]
    )

    if with_dense:
        dense_feats = np.zeros([batch_size, num_channels, *dense_shape], dtype=dtype)
        start = 0
        for i, inds in enumerate(batch_indices):
            for j, ind in enumerate(inds):
                dense_slice = (i, slice(None), *ind[:-1])
                dense_feats[dense_slice] = features[start + j]
            start += len(inds)
        sparse_dict["dense_feats"] = dense_feats
    batch_indices = np.concatenate(batch_indices, axis=0)
    sparse_dict["coords"] = batch_indices.astype(np.int32)

    return sparse_dict


def sparse_tensor_to_dense(
    ts_tensor,
    shape,
    num_channels=None,
    dtype=np.float16,
):
    ts_pt = ts_tensor.F[: ts_tensor.C.shape[0]]
    ts_coords = ts_tensor.C

    np_ts_pt = np.array(ts_pt.detach().cpu())
    np_ts_coords = np.array(ts_coords.detach().cpu())

    if num_channels is None:
        num_channels = np_ts_pt.shape[1]

    np_ts_pt = np_ts_pt[:, 0:num_channels]

    batch_size = np.max(np_ts_coords[:, 0]) - np.min(np_ts_coords[:, 0]) + 1

    dense_feats = np.zeros([batch_size, num_channels, *shape], dtype=dtype)

    for j, coord in enumerate(np_ts_coords):
        dense_slice = (coord[0], slice(None), *coord[1:])
        dense_feats[dense_slice] = np_ts_pt[j]

    return dense_feats


def dense_to_subm(feats, coords):
    # batch_size = feats.shape[0]
    # num_channels = feats.shape[1]

    mask = np.zeros(feats.shape, dtype=np.int32)

    for j, coord in enumerate(coords):
        dense_slice = (coord[0], slice(None), *coord[1:])
        mask[dense_slice] = 1

    subm_feats = feats * mask

    return subm_feats


def dense_pad(dense_feats_t, kernel_size):
    dense_feats_t = torch.nn.functional.pad(
        dense_feats_t,
        (0, kernel_size - 1, 0, kernel_size - 1, 0, kernel_size - 1),
        "constant",
        0,
    )
    return dense_feats_t

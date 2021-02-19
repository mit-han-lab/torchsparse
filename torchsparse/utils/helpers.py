from collections import Sequence

import numpy as np
import torch
from torchsparse import SparseTensor


def ravel_hash_vec(arr):
    assert arr.ndim == 2
    arr -= arr.min(0)
    arr = arr.astype(np.uint64, copy=False)
    arr_max = arr.max(0).astype(np.uint64) + 1

    keys = np.zeros(arr.shape[0], dtype=np.uint64)
    # Fortran style indexing
    for j in range(arr.shape[1] - 1):
        keys += arr[:, j]
        keys *= arr_max[j + 1]
    keys += arr[:, -1]
    return keys


def sparse_quantize(coords,
                    feats=None,
                    labels=None,
                    ignore_label=255,
                    return_index=False,
                    return_invs=False,
                    hash_type='ravel',
                    quantization_size=1):

    use_label = labels is not None
    use_feat = feats is not None
    if not use_label and not use_feat:
        return_index = True

    assert hash_type in [
        'ravel'
    ], "Invalid hash_type. Either ravel, or fnv allowed. You put hash_type=" + hash_type
    assert coords.ndim == 2
    if use_feat:
        assert feats.ndim == 2
        assert coords.shape[0] == feats.shape[0]
    if use_label:
        assert coords.shape[0] == len(labels)

    # Quantize the coordinates
    dimension = coords.shape[1]
    if isinstance(quantization_size, (Sequence, np.ndarray, torch.Tensor)):
        assert len(
            quantization_size
        ) == dimension, "Quantization size and coordinates size mismatch."
        quantization_size = [i for i in quantization_size]
    elif np.isscalar(quantization_size):  # Assume that it is a scalar
        quantization_size = [int(quantization_size) for i in range(dimension)]
    else:
        raise ValueError('Not supported type for quantization_size.')
    discrete_coords = np.floor(coords / np.array(quantization_size))

    # Hash function type
    key = ravel_hash_vec(discrete_coords)
    if use_label:
        _, inds, invs, counts = np.unique(key,
                                          return_index=True,
                                          return_inverse=True,
                                          return_counts=True)
        filtered_labels = labels[inds]
        filtered_labels[counts > 1] = ignore_label
        if return_invs:
            if return_index:
                return inds, filtered_labels, invs
            else:
                return discrete_coords[inds], feats[
                    inds], filtered_labels, invs
        else:
            if return_index:
                return inds, filtered_labels
            else:
                return discrete_coords[inds], feats[inds], filtered_labels

    else:
        _, inds, invs = np.unique(key, return_index=True, return_inverse=True)
        if return_invs:
            if return_index:
                return inds, invs
            else:
                if use_feat:
                    return discrete_coords[inds], feats[inds], invs
                else:
                    return discrete_coords[inds], invs
        else:
            if return_index:
                return inds
            else:
                if use_feat:
                    return discrete_coords[inds], feats[inds]
                else:
                    return discrete_coords[inds]


def sparse_collate(coords,
                   feats,
                   labels=None,
                   is_double=False,
                   coord_float=False):
    r"""Create a sparse tensor with batch indices C in `the documentation
    <https://stanfordvl.github.io/MinkowskiEngine/sparse_tensor.html>`_.

    Convert a set of coordinates and features into the batch coordinates and
    batch features.

    Args:
        coords (set of `torch.Tensor` or `numpy.ndarray`): a set of coordinates.

        feats (set of `torch.Tensor` or `numpy.ndarray`): a set of features.

        labels (set of `torch.Tensor` or `numpy.ndarray`): a set of labels
        associated to the inputs.

        is_double (`bool`): return double precision features if True. False by
        default.

    """
    use_label = False if labels is None else True
    coords_batch, feats_batch, labels_batch = [], [], []

    batch_id = 0
    for coord, feat in zip(coords, feats):
        if isinstance(coord, np.ndarray):
            coord = torch.from_numpy(coord)
        else:
            assert isinstance(
                coord, torch.Tensor
            ), "Coords must be of type numpy.ndarray or torch.Tensor"

        if not coord_float:
            coord = coord.int()
        else:
            coord = coord.float()

        if isinstance(feat, np.ndarray):
            feat = torch.from_numpy(feat)
        else:
            assert isinstance(
                feat, torch.Tensor
            ), "Features must be of type numpy.ndarray or torch.Tensor"
        feat = feat.double() if is_double else feat.float()

        # Batched coords
        num_points = coord.shape[0]

        if not coord_float:
            coords_batch.append(
                torch.cat((coord, torch.ones(num_points, 1).int() * batch_id),
                          1))
        else:
            coords_batch.append(
                torch.cat(
                    (coord, torch.ones(num_points, 1).float() * batch_id), 1))

        # Features
        feats_batch.append(feat)

        # Labels
        if use_label:
            label = labels[batch_id]
            if isinstance(label, np.ndarray):
                label = torch.from_numpy(label)
            else:
                assert isinstance(
                    label, torch.Tensor
                ), "labels must be of type numpy.ndarray or torch.Tensor"
            labels_batch.append(label)

        batch_id += 1

    # Concatenate all lists
    if not coord_float:
        coords_batch = torch.cat(coords_batch, 0).int()
    else:
        coords_batch = torch.cat(coords_batch, 0).float()
    feats_batch = torch.cat(feats_batch, 0)
    if use_label:
        labels_batch = torch.cat(labels_batch, 0)
        return coords_batch, feats_batch, labels_batch
    else:
        return coords_batch, feats_batch


def sparse_collate_tensors(sparse_tensors):
    coords, feats = sparse_collate([x.C for x in sparse_tensors],
                                   [x.F for x in sparse_tensors])
    return SparseTensor(feats, coords, sparse_tensors[0].s)


def sparse_collate_fn(batch):
    if isinstance(batch[0], dict):
        batch_size = batch.__len__()
        ans_dict = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], SparseTensor):
                ans_dict[key] = sparse_collate_tensors(
                    [sample[key] for sample in batch])
            elif isinstance(batch[0][key], np.ndarray):
                ans_dict[key] = torch.stack(
                    [torch.from_numpy(sample[key]) for sample in batch],
                    axis=0)
            elif isinstance(batch[0][key], torch.Tensor):
                ans_dict[key] = torch.stack([sample[key] for sample in batch],
                                            axis=0)
            elif isinstance(batch[0][key], dict):
                ans_dict[key] = sparse_collate_fn(
                    [sample[key] for sample in batch])
            else:
                ans_dict[key] = [sample[key] for sample in batch]
        return ans_dict
    else:
        batch_size = batch.__len__()
        ans_dict = tuple()
        for i in range(len(batch[0])):
            key = batch[0][i]
            if isinstance(key, SparseTensor):
                ans_dict += sparse_collate_tensors(
                    [sample[i] for sample in batch]),
            elif isinstance(key, np.ndarray):
                ans_dict += torch.stack(
                    [torch.from_numpy(sample[i]) for sample in batch], axis=0),
            elif isinstance(key, torch.Tensor):
                ans_dict += torch.stack([sample[i] for sample in batch],
                                        axis=0),
            elif isinstance(key, dict):
                ans_dict += sparse_collate_fn([sample[i] for sample in batch]),
            else:
                ans_dict += [sample[i] for sample in batch],
        return ans_dict

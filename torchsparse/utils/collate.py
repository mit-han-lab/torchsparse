from typing import Any, List

import numpy as np
import torch

from torchsparse import SparseTensor

__all__ = ['sparse_collate', 'sparse_collate_fn']


def sparse_collate(inputs: List[SparseTensor]) -> SparseTensor:
    coords, feats = [], []
    stride = inputs[0].stride

    for k, x in enumerate(inputs):
        if isinstance(x.coords, np.ndarray):
            x.coords = torch.tensor(x.coords)
        if isinstance(x.feats, np.ndarray):
            x.feats = torch.tensor(x.feats)

        assert isinstance(x.coords, torch.Tensor), type(x.coords)
        assert isinstance(x.feats, torch.Tensor), type(x.feats)
        assert x.stride == stride, (x.stride, stride)

        input_size = x.coords.shape[0]
        batch = torch.full((input_size, 1),
                           k,
                           device=x.coords.device,
                           dtype=torch.int)

        coords.append(torch.cat((x.coords, batch), dim=1))
        feats.append(x.feats)

    coords = torch.cat(coords, dim=0)
    feats = torch.cat(feats, dim=0)
    output = SparseTensor(coords=coords, feats=feats, stride=stride)
    return output


def sparse_collate_fn(inputs: List[Any]) -> Any:
    if isinstance(inputs[0], dict):
        output = {}
        for name in inputs[0].keys():
            if isinstance(inputs[0][name], dict):
                output[name] = sparse_collate_fn(
                    [input[name] for input in inputs])
            elif isinstance(inputs[0][name], np.ndarray):
                output[name] = torch.stack(
                    [torch.tensor(input[name]) for input in inputs], dim=0)
            elif isinstance(inputs[0][name], torch.Tensor):
                output[name] = torch.stack([input[name] for input in inputs],
                                           dim=0)
            elif isinstance(inputs[0][name], SparseTensor):
                output[name] = sparse_collate([input[name] for input in inputs])
            else:
                output[name] = [input[name] for input in inputs]
        return output
    else:
        return inputs

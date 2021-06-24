from typing import List

import torch

from torchsparse.tensors import SparseTensor

__all__ = ['cat']


def cat(inputs: List[SparseTensor]) -> SparseTensor:
    coords, stride = inputs[0].coords, inputs[0].stride
    feats = torch.cat([inputs.feats for inputs in inputs], dim=1)
    outputs = SparseTensor(feats, coords, stride)
    outputs.cmaps = inputs[0].cmaps
    outputs.kmaps = inputs[0].kmaps
    return outputs

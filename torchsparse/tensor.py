from typing import Any, Dict, Tuple, Union, Optional, List

import torch

from torchsparse.utils import make_ntuple, to_dense
from torchsparse.utils.tensor_cache import (
    TensorCache,
    TensorCacheMode,
    get_global_tensor_cache,
    set_global_tensor_cache,
    get_tensor_cache_mode,
)

__all__ = ["SparseTensor"]

_allow_negative_coordinates = False


def get_allow_negative_coordinates():
    global _allow_negative_coordinates
    return _allow_negative_coordinates


def set_allow_negative_coordinates(allow_negative_coordinates):
    global _allow_negative_coordinates
    _allow_negative_coordinates = allow_negative_coordinates


class SparseTensor:
    def __init__(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor,
        stride: Union[int, Tuple[int, ...]] = 1,
        spatial_range: Union[int, Tuple[int, ...]] = None,
    ) -> None:
        self.feats = feats
        self.coords = coords
        self.stride = make_ntuple(stride, ndim=3)
        if spatial_range is None:
            self.spatial_range = None
        else:
            self.spatial_range = make_ntuple(spatial_range, ndim=len(spatial_range))

        if get_tensor_cache_mode() == TensorCacheMode.GLOBAL_TENSOR_CACHE:
            _caches = get_global_tensor_cache()
            if _caches is None:
                _caches = TensorCache()
                set_global_tensor_cache(_caches)
            self._caches = _caches
        else:
            self._caches = TensorCache()

    @property
    def F(self) -> torch.Tensor:
        return self.feats

    @F.setter
    def F(self, feats: torch.Tensor) -> None:
        self.feats = feats

    @property
    def C(self) -> torch.Tensor:
        return self.coords

    @C.setter
    def C(self, coords: torch.Tensor) -> None:
        self.coords = coords

    @property
    def s(self) -> Tuple[int, ...]:
        return self.stride

    @s.setter
    def s(self, stride: Union[int, Tuple[int, ...]]) -> None:
        self.stride = make_ntuple(stride, ndim=3)

    def cpu(self):
        self.coords = self.coords.cpu()
        self.feats = self.feats.cpu()
        return self

    def cuda(self):
        self.coords = self.coords.cuda()
        self.feats = self.feats.cuda()
        return self

    def half(self):
        self.feats = self.feats.half()
        return self

    def detach(self):
        self.coords = self.coords.detach()
        self.feats = self.feats.detach()
        return self

    def to(self, device, non_blocking: bool = True):
        self.coords = self.coords.to(device, non_blocking=non_blocking)
        self.feats = self.feats.to(device, non_blocking=non_blocking)
        return self

    def dense(self):
        assert self.spatial_range is not None
        return to_dense(self.feats, self.coords, self.spatial_range)

    def __add__(self, other):
        output = SparseTensor(
            coords=self.coords,
            feats=self.feats + other.feats,
            stride=self.stride,
            spatial_range=self.spatial_range,
        )
        output._caches = self._caches
        return output
    
class PointTensor:
    def __init__(self, feats, coords, idx_query=None, weights=None):
        self.F = feats
        self.C = coords
        self.idx_query = idx_query if idx_query is not None else {}
        self.weights = weights if weights is not None else {}
        self.additional_features = {}
        self.additional_features['idx_query'] = {}
        self.additional_features['counts'] = {}

    def cuda(self):
        self.F = self.F.cuda()
        self.C = self.C.cuda()
        return self

    def detach(self):
        self.F = self.F.detach()
        self.C = self.C.detach()
        return self

    def to(self, device, non_blocking=True):
        self.F = self.F.to(device, non_blocking=non_blocking)
        self.C = self.C.to(device, non_blocking=non_blocking)
        return self

    def __add__(self, other):
        tensor = PointTensor(self.F + other.F, self.C, self.idx_query,
                             self.weights)
        tensor.additional_features = self.additional_features
        return tensor


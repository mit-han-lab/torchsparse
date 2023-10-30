from typing import Any, Dict, Tuple, Union
from enum import Enum
import copy


class TensorCacheMode(Enum):
    SEPARATE_TENSOR_CACHE = 0
    GLOBAL_TENSOR_CACHE = 1


_tensor_cache_mode = TensorCacheMode.SEPARATE_TENSOR_CACHE
_global_tensor_cache = None


def set_tensor_cache_mode(mode: TensorCacheMode):
    r"""
    _tensor_cache_mode is set SEPARATE_TENSOR_CACHE by default
    if _tensor_cache_mode is set to GLOBAL_TENSOR_CACHE
    the _global_tensor_cache must be cleared after each forward/backward
    """
    assert isinstance(
        mode, TensorCacheMode
    ), f"Input must be an instance of TensorCacheMode"
    global _tensor_cache_mode
    _tensor_cache_mode = mode


def get_tensor_cache_mode() -> TensorCacheMode:
    global _tensor_cache_mode
    return copy.deepcopy(_tensor_cache_mode)


class TensorCache:
    def __init__(
        self,
    ) -> None:
        self.cmaps: Dict[Tuple[int, ...], Tuple[torch.Tensor, Tuple[int, ...]]] = {}
        self.kmaps: Dict[Tuple[Any, ...], Any] = {}
        self.hashmaps: Dict[Tuple[int, ...], Tuple[Any, ...]] = {}


def get_global_tensor_cache():
    global _global_tensor_cache
    return _global_tensor_cache


def set_global_tensor_cache(tensor_cache):
    global _global_tensor_cache
    _global_tensor_cache = tensor_cache


def clear_global_tensor_cache():
    r"""
    if _tensor_cache_mode is set to GLOBAL_TENSOR_CACHE
    the _global_tensor_cache must be cleared after each forward/backward
    """
    global _global_tensor_cache
    _global_tensor_cache = None

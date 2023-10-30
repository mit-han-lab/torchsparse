from enum import Enum

_global_kmap_mode = "hashmap_on_the_fly"  # or "hashmap"
_global_downsample_mode = "spconv"  # or "minkowski"


def get_kmap_mode():
    global _global_kmap_mode
    return _global_kmap_mode


def set_kmap_mode(kmap_mode: str):
    global _global_kmap_mode
    if kmap_mode in ["hashmap_on_the_fly", "hashmap"]:
        _global_kmap_mode = kmap_mode
    else:
        assert (
            0
        ), f'Unsupport kmap_mode: {kmap_mode}. Please set kmap_mode to "hashmap_on_the_fly" or "hashmap".'


def get_downsample_mode():
    global _global_downsample_mode
    return _global_downsample_mode


def set_downsample_mode(downsample_mode: str):
    global _global_downsample_mode
    if downsample_mode in ["spconv", "minkowski"]:
        _global_downsample_mode = downsample_mode
    else:
        assert (
            0
        ), f'Unsupport downsample_mode {downsample_mode}. Please set downsample_mode to "spconv" or "minkowski".'


class ConvMode(Enum):
    mode0 = 0  # split=0 fwd & split=3 bwd
    mode1 = 1  # split=1 fwd & split=3 bwd
    mode2 = 2  # split=3 fwd & split=3 bwd


_global_conv_mode = ConvMode.mode0


def get_conv_mode():
    global _global_conv_mode
    return _global_conv_mode


def set_conv_mode(conv_mode):
    global _global_conv_mode
    if isinstance(conv_mode, int):
        if conv_mode == 0:
            _global_conv_mode = ConvMode.mode0
        elif conv_mode == 1:
            _global_conv_mode = ConvMode.mode1
        elif conv_mode == 2:
            _global_conv_mode = ConvMode.mode2
        else:
            assert 0, f"Undefined conv_mode:{conv_mode}"
    elif isinstance(conv_mode, ConvMode):
        _global_conv_mode = conv_mode
    else:
        assert 0, f"Unsupport conv_mode input type"

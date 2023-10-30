from typing import Any, Dict, Tuple, Union
from enum import Enum
from .utils import AttributeDict
from .conv_mode import ConvMode, get_kmap_mode, get_downsample_mode


class Dataflow(Enum):
    ImplicitGEMM = 0
    GatherScatter = 1
    FetchOnDemand = 2
    CodedCSR = 3


_global_conv_config = None
_default_conv_config = AttributeDict(
    [
        ("dataflow", Dataflow.ImplicitGEMM),
        ("ifsort", False),
        ("kmap_mode", "hashmap_on_the_fly"),
        ("downsample_mode", "spconv"),
        ("split_mask_num", 1),
        ("split_mask_num_bwd", 3),
        ("epsilon", 0.0),
        ("mm_thresh", 0),
        ("FOD_fusion", True),
    ]
)


def keys_check(conv_config):
    flag = False
    if "dataflow" not in conv_config:
        flag = True
        conv_config["dataflow"] = _default_conv_config["dataflow"]
    if "ifsort" not in conv_config:
        flag = True
        conv_config["ifsort"] = _default_conv_config["ifsort"]
    if "kmap_mode" not in conv_config:
        flag = True
        conv_config["kmap_mode"] = _default_conv_config["kmap_mode"]
    if "downsample_mode" not in conv_config:
        flag = True
        conv_config["downsample_mode"] = _default_conv_config["downsample_mode"]
    if "split_mask_num" not in conv_config:
        flag = True
        conv_config["split_mask_num"] = _default_conv_config["split_mask_num"]
    if "split_mask_num_bwd" not in conv_config:
        flag = True
        conv_config["split_mask_num_bwd"] = _default_conv_config["split_mask_num_bwd"]
    if "epsilon" not in conv_config:
        flag = True
        conv_config["epsilon"] = _default_conv_config["epsilon"]
    if "mm_thresh" not in conv_config:
        flag = True
        conv_config["mm_thresh"] = _default_conv_config["mm_thresh"]
    if "FOD_fusion" not in conv_config:
        flag = True
        conv_config["FOD_fusion"] = _default_conv_config["FOD_fusion"]
    if flag == True:
        print(
            "Warning: Missing fields for ConvConfig. Use default configs for these fields."
        )


def get_global_conv_config():
    global _global_conv_config
    return _global_conv_config


def set_global_conv_config(conv_config):
    global _global_conv_config
    keys_check(conv_config)
    _global_conv_config = conv_config


def clear_global_conv_config():
    global _global_conv_config
    _global_conv_config = None


def get_default_conv_config(
    conv_mode: ConvMode = ConvMode.mode0, training: bool = False
):
    config = _default_conv_config
    # if training:
    #     config.ifsort = True
    if conv_mode == ConvMode.mode0:
        pass
    elif conv_mode == ConvMode.mode1:
        config.ifsort = True
    elif conv_mode == ConvMode.mode2:
        config.ifsort = True
        config.split_mask_num = 3
    return config

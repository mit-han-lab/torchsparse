from typing import Dict, Tuple, Union
import math
import numpy as np
import torch

import torchsparse.backend
from torchsparse.utils import make_ntuple, make_tensor, make_divisible

from .func import *

from ..conv_config import *

__all__ = ["build_kernel_map", "transpose_kernel_map"]

cta_M = 128
cta_M_wgrad = 64


def build_kernel_map(
    _coords: torch.Tensor,
    input_node_num: int,
    kernel_size: Union[int, Tuple[int, ...]] = 2,
    stride: Union[int, Tuple[int, ...]] = 2,
    padding: Union[int, Tuple[int, ...]] = 0,
    hashmap_keys: torch.Tensor = None,
    hashmap_vals: torch.Tensor = None,
    spatial_range: int = None,
    mode="hashmap",
    dataflow=Dataflow.ImplicitGEMM,
    downsample_mode="spconv",
    training: bool = False,
    ifsort: bool = False,
    generative: bool = False,
    split_mask_num: int = 1,
    split_mask_num_bwd: int = 1,
) -> Dict:
    from torchsparse.nn import functional as F

    kmap = dict(
        [
            ("out_in_map", None),
            ("coords", None),
            ("sizes", None),
            ("reorder_out_in_map", None),
            ("reduced_sorted_mask", None),
            ("reorder_loc", None),
            ("nbmaps", None),
            ("nbsizes", None),
            ("input_mask", None),
            ("output_mask", None),
            ("hashmap_keys", hashmap_keys),
            ("hashmap_vals", hashmap_vals),
            ("spatial_range", spatial_range),
            # [Fetch-on-Demand]: (quantified) neighbor addresses
            ("nbaddrs", None),
            ("qnbaddrs", None),
            # [Fetch-on-Demand]: quantified mapsize
            ("qmapsize", None),
        ]
    )

    stride = make_ntuple(stride, ndim=3)
    kernel_size = make_ntuple(kernel_size, ndim=3)
    padding = make_ntuple(padding, ndim=3)
    if spatial_range is not None:
        new_spatial_range = [0, 0, 0]
        for i in range(len(new_spatial_range)):
            new_spatial_range[i] = (
                spatial_range[i + 1] + 2 * padding[i] - (kernel_size[i] - 1) - 1
            ) // stride[i] + 1
        new_spatial_range = spatial_range[:1] + tuple(new_spatial_range)
    else:
        new_spatial_range = None
    subm = not (any(s > 1 for s in stride))
    stride = make_tensor(stride, dtype=torch.int, device=_coords.device)
    padding = make_tensor(padding, dtype=torch.int, device=_coords.device)
    kernel_size = make_tensor(kernel_size, dtype=torch.int, device=_coords.device)

    if mode == "hashmap_on_the_fly":
        if generative:
            raise ValueError(
                f"Unsupported kmap_mode: {mode} for generative convolution (please switch to kmap_mode=hashmap)."
            )
        if dataflow == Dataflow.ImplicitGEMM:
            kmap = build_kmap_implicit_GEMM_hashmap_on_the_fly(
                kmap,
                input_node_num,
                _coords,
                kernel_size,
                stride,
                padding=padding,
                spatial_range=new_spatial_range,
                cta_M=cta_M,
                subm=subm,
                ifsort=ifsort,
                split_mask_num=split_mask_num,
            )

        elif dataflow == Dataflow.GatherScatter:
            kmap = build_kmap_Gather_Scatter_hashmap_on_the_fly(
                kmap,
                input_node_num,
                _coords,
                kernel_size,
                stride,
                padding=padding,
                spatial_range=new_spatial_range,
                cta_M=cta_M,
                subm=subm,
            )

        elif dataflow == Dataflow.FetchOnDemand:
            kmap = build_kmap_Fetch_on_Demand_hashmap_on_the_fly(
                kmap,
                input_node_num,
                _coords,
                kernel_size,
                stride,
                padding=padding,
                spatial_range=new_spatial_range,
                cta_M=cta_M,
                subm=subm,
            )

        else:
            raise ValueError(
                "[Build kernel map] unsupported dataflow: {}".format(dataflow)
            )

    elif mode == "hashmap":

        if dataflow == Dataflow.ImplicitGEMM:
            kmap = build_kmap_implicit_GEMM_hashmap(
                kmap,
                input_node_num,
                _coords,
                kernel_size,
                stride,
                padding=padding,
                spatial_range=new_spatial_range,
                cta_M=cta_M,
                subm=subm,
                ifsort=ifsort,
                downsample_mode=downsample_mode,
                generative=generative,
                split_mask_num=split_mask_num,
            )

        elif dataflow == Dataflow.GatherScatter:
            kmap = build_kmap_Gather_Scatter_hashmap(
                kmap,
                input_node_num,
                _coords,
                kernel_size,
                stride,
                padding=padding,
                spatial_range=new_spatial_range,
                cta_M=cta_M,
                subm=subm,
                downsample_mode=downsample_mode,
                generative=generative,
            )

        elif dataflow == Dataflow.FetchOnDemand:
            kmap = build_kmap_Fetch_on_Demand_hashmap(
                kmap,
                input_node_num,
                _coords,
                kernel_size,
                stride,
                padding=padding,
                spatial_range=new_spatial_range,
                cta_M=cta_M,
                subm=subm,
                downsample_mode=downsample_mode,
                generative=generative,
            )

        else:
            raise ValueError(
                "[Build kernel map] unsupported dataflow: {}".format(dataflow)
            )

    elif mode == "grid":
        assert 0, "grid mode is temporarily deprecated."

    else:
        raise ValueError("[Build kernel map] unknown mode: {}".format(mode))

    if dataflow == Dataflow.ImplicitGEMM:
        if training:
            out_in_map_bwd = F.convert_transposed_out_in_map(
                kmap["out_in_map"], make_divisible(kmap["sizes"][0], cta_M)
            )
            bitmask_bwd = torchsparse.backend.derive_bitmask_from_out_in_map(
                out_in_map_bwd, split_mask_num_bwd, kmap["sizes"][0]
            )
            sorted_mask_bwd, reorder_loc_bwd = torch.sort(bitmask_bwd, descending=True)
            reorder_loc_bwd = reorder_loc_bwd.to(torch.int32)
            reorder_out_in_map_bwd = torchsparse.backend.reorder_out_in_map_cuda(
                out_in_map_bwd, reorder_loc_bwd
            )
            reduced_sorted_mask_bwd_wgrad = torchsparse.backend.reduce_bitmask_cuda(
                sorted_mask_bwd, cta_M_wgrad
            )
            reduced_sorted_mask_bwd_dgrad = torchsparse.backend.reduce_bitmask_cuda(
                sorted_mask_bwd, cta_M
            )
        else:
            out_in_map_bwd = None
            reorder_out_in_map_bwd = None
            reduced_sorted_mask_bwd_wgrad = None
            reduced_sorted_mask_bwd_dgrad = None
            reorder_loc_bwd = None
        kmap["out_in_map_bwd"] = out_in_map_bwd
        kmap["reorder_out_in_map_bwd"] = reorder_out_in_map_bwd
        kmap["reduced_sorted_mask_bwd_wgrad"] = reduced_sorted_mask_bwd_wgrad
        kmap["reduced_sorted_mask_bwd_dgrad"] = reduced_sorted_mask_bwd_dgrad
        kmap["reorder_loc_bwd"] = reorder_loc_bwd
    return kmap


def transpose_kernel_map(
    kmap: Dict,
    ifsort: bool = False,
    training: bool = False,
    split_mask_num: int = 1,
    split_mask_num_bwd: int = 1,
) -> Dict:
    from torchsparse.nn import functional as F

    out_in_map = F.convert_transposed_out_in_map(
        kmap["out_in_map"], make_divisible(kmap["sizes"][0], cta_M)
    )

    if ifsort:
        if training:
            out_in_map_bwd = kmap["out_in_map"]
            reorder_out_in_map_bwd = kmap["reorder_out_in_map"]
            reorder_loc_bwd = kmap["reorder_loc"]
            sorted_mask_bwd = kmap["sorted_mask"]
            reduced_sorted_mask_bwd_wgrad = torchsparse.backend.reduce_bitmask_cuda(
                sorted_mask_bwd, cta_M_wgrad
            )
            reduced_sorted_mask_bwd_dgrad = torchsparse.backend.reduce_bitmask_cuda(
                sorted_mask_bwd, cta_M
            )
            kmap["out_in_map_bwd_t"] = out_in_map_bwd
            kmap["reorder_out_in_map_bwd_t"] = reorder_out_in_map_bwd
            kmap["reduced_sorted_mask_bwd_wgrad_t"] = reduced_sorted_mask_bwd_wgrad
            kmap["reduced_sorted_mask_bwd_dgrad_t"] = reduced_sorted_mask_bwd_dgrad
            kmap["reorder_loc_bwd_t"] = reorder_loc_bwd
        else:
            kmap["out_in_map_bwd_t"] = None
            kmap["reorder_out_in_map_bwd_t"] = None
            kmap["reduced_sorted_mask_bwd_wgrad_t"] = None
            kmap["reduced_sorted_mask_bwd_dgrad_t"] = None
            kmap["reorder_loc_bwd_t"] = None

        bitmask = torchsparse.backend.derive_bitmask_from_out_in_map(
            out_in_map, split_mask_num, kmap["sizes"][0]
        )
        sorted_mask, reorder_loc = torch.sort(bitmask, descending=True)
        reorder_loc = reorder_loc.to(torch.int32)
        reorder_out_in_map = torchsparse.backend.reorder_out_in_map_cuda(
            out_in_map, reorder_loc
        )
        reduced_sorted_mask = torchsparse.backend.reduce_bitmask_cuda(
            sorted_mask, cta_M
        )
        kmap["reorder_out_in_map_t"] = reorder_out_in_map
        kmap["reduced_sorted_mask_t"] = reduced_sorted_mask
        kmap["reorder_loc_t"] = reorder_loc
    else:
        if training:
            out_in_map_bwd = kmap["out_in_map"]
            bitmask_bwd = torchsparse.backend.derive_bitmask_from_out_in_map(
                out_in_map_bwd, split_mask_num_bwd, kmap["sizes"][1]
            )
            sorted_mask_bwd, reorder_loc_bwd = torch.sort(bitmask_bwd, descending=True)
            reorder_loc_bwd = reorder_loc_bwd.to(torch.int32)
            reorder_out_in_map_bwd = torchsparse.backend.reorder_out_in_map_cuda(
                out_in_map_bwd, reorder_loc_bwd
            )
            reduced_sorted_mask_bwd_wgrad = torchsparse.backend.reduce_bitmask_cuda(
                sorted_mask_bwd, cta_M_wgrad
            )
            reduced_sorted_mask_bwd_dgrad = torchsparse.backend.reduce_bitmask_cuda(
                sorted_mask_bwd, cta_M
            )
            kmap["out_in_map_bwd_t"] = out_in_map_bwd
            kmap["reorder_out_in_map_bwd_t"] = reorder_out_in_map_bwd
            kmap["reduced_sorted_mask_bwd_wgrad_t"] = reduced_sorted_mask_bwd_wgrad
            kmap["reduced_sorted_mask_bwd_dgrad_t"] = reduced_sorted_mask_bwd_dgrad
            kmap["reorder_loc_bwd_t"] = reorder_loc_bwd
        else:
            kmap["out_in_map_bwd_t"] = None
            kmap["reorder_out_in_map_bwd_t"] = None
            kmap["reduced_sorted_mask_bwd_wgrad_t"] = None
            kmap["reduced_sorted_mask_bwd_dgrad_t"] = None
            kmap["reorder_loc_bwd_t"] = None
        kmap["reorder_out_in_map_t"] = None
        kmap["reduced_sorted_mask_t"] = None
        kmap["reorder_loc_t"] = None

    kmap["out_in_map_t"] = out_in_map

    return kmap

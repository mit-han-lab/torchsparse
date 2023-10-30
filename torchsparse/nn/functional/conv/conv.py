from typing import List, Dict, Optional, Tuple, Union

# import numpy as np
import torch

import torchsparse
from torchsparse import SparseTensor
from torchsparse.utils import make_ntuple

from .func import *

__all__ = ["conv3d"]


def conv3d(
    input: SparseTensor,
    weight: torch.Tensor,
    kernel_size: Union[int, List[int], Tuple[int, ...]],
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, List[int], Tuple[int, ...]] = 1,
    padding: Union[int, Tuple[int, ...]] = 0,
    dilation: Union[int, Tuple[int, ...]] = 1,
    config: Dict = None,
    transposed: bool = False,
    generative: bool = False,
    training: bool = False,
) -> SparseTensor:
    from torchsparse.nn import functional as F

    feats, coords = input.feats, input.coords
    kernel_size = make_ntuple(kernel_size, ndim=3)
    # kernel_volume = np.prod(kernel_size)
    stride = make_ntuple(stride, ndim=3)
    dilation = make_ntuple(dilation, ndim=3)

    conv_mode = F.get_conv_mode()
    if config is None:
        config = F.conv_config.get_global_conv_config()
        if config is None:
            config = F.conv_config.get_default_conv_config(
                conv_mode=conv_mode, training=training
            )

    # TODO: Deal with kernel volume > 32. (Split mask or unsort)

    dataflow = config.dataflow
    kmap_mode = config.kmap_mode

    if dataflow == F.Dataflow.ImplicitGEMM:
        ConvolutionFunction = ImplicitGEMMConvolutionFuntion
    elif dataflow == F.Dataflow.GatherScatter:
        ConvolutionFunction = GatherScatterConvolutionFuntion
        config.ifsort = False
    elif dataflow == F.Dataflow.FetchOnDemand:
        ConvolutionFunction = FetchOnDemandConvolutionFuntion
        config.ifsort = False
    elif (
        dataflow == F.Dataflow.CodedCSR
    ):  # Placeholder for PCEngine integration. Mode name can be modified.
        config.ifsort = False
        assert 0, "CodedCSR has not been integrated."
    else:
        raise ValueError("unsupported dataflow: {}".format(dataflow))

    if kernel_size == (1, 1, 1) and stride == (1, 1, 1) and dilation == (1, 1, 1):
        feats = feats.matmul(weight)
        if bias is not None:
            feats += bias
        output = SparseTensor(
            coords=coords,
            feats=feats,
            stride=input.stride,
            spatial_range=input.spatial_range,
        )
    elif not transposed:
        kmap = input._caches.kmaps.get((input.stride, kernel_size, stride, dilation))

        if kmap_mode != "hashmap_on_the_fly":
            hashmap = input._caches.hashmaps.get(input.stride)
        else:
            hashmap = input._caches.hashmaps.get(
                tuple(input.stride[k] * stride[k] for k in range(3))
            )
        if hashmap is None:
            hashmap_keys, hashmap_vals = None, None
        else:
            hashmap_keys, hashmap_vals = hashmap

        spatial_range = input.spatial_range

        if kmap is None:
            kmap = F.build_kernel_map(
                coords,
                feats.shape[0],
                kernel_size,
                stride,
                padding,
                hashmap_keys,
                hashmap_vals,
                spatial_range,
                kmap_mode,
                dataflow,
                downsample_mode=config.downsample_mode,
                training=training,
                ifsort=config.ifsort,
                split_mask_num=config.split_mask_num,
                split_mask_num_bwd=config.split_mask_num_bwd,
            )

            hashmap = [kmap["hashmap_keys"], kmap["hashmap_vals"]]

            input._caches.kmaps[(input.stride, kernel_size, stride, dilation)] = kmap
            input._caches.hashmaps[input.stride] = hashmap

        feats = ConvolutionFunction.apply(
            feats,
            weight,
            kmap,
            config,
            transposed,
        )

        if bias is not None:
            feats += bias
        output = SparseTensor(
            coords=kmap["coords"],
            feats=feats,
            stride=tuple(input.stride[k] * stride[k] for k in range(3)),
            spatial_range=kmap["spatial_range"],
        )
    else:
        tensor_stride = tuple(input.stride[k] // stride[k] for k in range(3))
        if not generative:
            kmap = input._caches.kmaps.get(
                (tensor_stride, kernel_size, stride, dilation)
            )

            kmap = F.transpose_kernel_map(
                kmap,
                config.ifsort,
                training=training,
                split_mask_num=config.split_mask_num,
                split_mask_num_bwd=config.split_mask_num_bwd,
            )

            feats = ConvolutionFunction.apply(
                feats,
                weight,
                kmap,
                config,
                transposed,
            )

            if bias is not None:
                feats += bias
            output = SparseTensor(
                coords=input._caches.cmaps[tensor_stride][0],
                feats=feats,
                stride=tensor_stride,
                spatial_range=input._caches.cmaps[tensor_stride][1],
            )
        else:
            hashmap_keys, hashmap_vals = None, None

            spatial_range = input.spatial_range
            kmap = F.build_kernel_map(
                coords,
                feats.shape[0],
                kernel_size,
                stride,
                padding,
                hashmap_keys,
                hashmap_vals,
                spatial_range,
                kmap_mode,
                dataflow,
                downsample_mode=config.downsample_mode,
                training=training,
                ifsort=config.ifsort,
                generative=generative,
            )
            # generate output: logically forced to be not transposed
            feats = ConvolutionFunction.apply(
                feats,
                weight,
                kmap,
                config,
                False,
            )
            if bias is not None:
                feats += bias
            input._caches.cmaps[tensor_stride] = (
                kmap["coords"],
                kmap.get("spatial_range"),
            )
            output = SparseTensor(
                coords=input._caches.cmaps[tensor_stride][0],
                feats=feats,
                stride=tensor_stride,
                spatial_range=input._caches.cmaps[tensor_stride][1],
            )
            hashmap = [kmap["hashmap_keys"], kmap["hashmap_vals"]]
            input._caches.kmaps = dict()  # new_kmap
            input._caches.hashmaps = dict()

    output._caches = input._caches
    output._caches.cmaps.setdefault(
        output.stride, (output.coords, output.spatial_range)
    )
    return output

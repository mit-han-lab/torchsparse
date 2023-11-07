from typing import Any, Dict, Tuple, Union, Optional, List

import numpy as np
import torch
from torch import nn

import torchsparse
from torchsparse import nn as spnn
from torchsparse.nn import functional as F
from torchsparse.utils import make_ntuple

from .test_utils import *

__all__ = ["test_single_layer_convolution_forward"]


class TestSparseConv(nn.Module):
    def __init__(
        self,
        num_layers,
        shape,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        device,
    ):
        super().__init__()
        layers = [
            spnn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
            )
        ]

        for i in range(1, num_layers):
            layers.append(
                spnn.Conv3d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                )
            )
        self.net = nn.Sequential(
            *layers,
        ).to(device)
        self.shape = shape

    def forward(self, feats, coords):
        coords = coords.int()
        ts_tensor = torchsparse.SparseTensor(feats, coords)
        return self.net(ts_tensor)


class TestTorchConv(nn.Module):
    def __init__(
        self,
        num_layers,
        shape,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        device,
    ):
        super().__init__()
        layers = [
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                bias=False,
            )
        ]

        for i in range(1, num_layers):
            layers.append(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    bias=False,
                )
            )
        self.net = nn.Sequential(
            *layers,
        ).to(device)
        self.shape = shape

    def forward(self, x):
        return self.net(x)


def test_single_layer_convolution_forward(
    batch_size: int = 1,
    shape: Union[int, Tuple[int, ...]] = 5,
    num_points: int = 20,
    IC: int = 16,
    OC: int = 32,
    kernel_size: int = 3,
    stride: int = 1,
    device="cuda:0",
    is_half=True,
):

    np.random.seed(0)
    torch.manual_seed(0)

    shape = make_ntuple(shape, ndim=3)
    if num_points > np.prod(shape):
        print("Warning: num_points exceeds coords range!")
        print("         reduce num_points to %d!" % np.prod(shape))
        num_points = np.prod(shape)
    num_points = [num_points] * batch_size

    if kernel_size % 2 == 0:
        layer_padding = 0
    else:
        layer_padding = (kernel_size - 1) // 2

    model = TestSparseConv(
        num_layers=1,
        shape=shape,
        in_channels=IC,
        out_channels=OC,
        kernel_size=kernel_size,
        stride=stride,
        padding=layer_padding,
        dilation=1,
        device=device,
    )

    ref_model = TestTorchConv(
        num_layers=1,
        shape=shape,
        in_channels=IC,
        out_channels=OC,
        kernel_size=kernel_size,
        stride=stride,
        padding=layer_padding,
        dilation=1,
        device=device,
    )

    if is_half:
        torch_dtype = torch.float16
        np_dtype = np.float16
        model.half()
        ref_model.half()

    else:
        torch_dtype = torch.float32
        np_dtype = np.float32

    sparse_dict = generate_feature_map(shape, num_points, IC, dtype=np_dtype)

    feats = np.ascontiguousarray(sparse_dict["feats"])
    coords = np.ascontiguousarray(sparse_dict["coords"][:, [3, 0, 1, 2]])  # batch first
    dense_feats = sparse_dict["dense_feats"]

    # print(feats)
    # print(coords)
    # print(dense_feats)

    coords_t = torch.from_numpy(coords).int().to(device)
    feats_t = torch.from_numpy(feats).to(torch_dtype).to(device)
    dense_feats_t = torch.from_numpy(dense_feats).to(torch_dtype).to(device)

    filters = np.random.uniform(
        -1, 1, size=[kernel_size, kernel_size, kernel_size, IC, OC]
    ).astype(np_dtype)
    filters_t = torch.from_numpy(filters).to(torch_dtype).to(device)

    if kernel_size % 2 == 1:
        ref_model.net[0].weight.data[:] = filters_t.permute(4, 3, 2, 1, 0).contiguous()
    else:
        ref_model.net[0].weight.data[:] = filters_t.permute(4, 3, 0, 1, 2).contiguous()

    model.net[0].kernel.data[:] = filters_t.reshape(-1, IC, OC)

    if kernel_size % 2 == 0:  # manually pad
        dense_feats_t = dense_pad(dense_feats_t, kernel_size)

    ref_out = ref_model(dense_feats_t)
    out = model(feats_t, coords_t)

    ts_coords = out.C
    ts_coords_np = np.array(ts_coords.detach().cpu())

    ref_out_np = ref_out.detach().cpu().numpy()
    ref_out_subm_np = dense_to_subm(ref_out_np, ts_coords_np)

    out_dense_np = sparse_tensor_to_dense(out, ref_out_np.shape[2:], OC, dtype=np_dtype)

    # print(out.C)
    # print(out.F)

    # print(ref_out_np)
    # print(out_dense_np)
    mean_adiff = np.sum(np.abs(out_dense_np - ref_out_subm_np)) / ts_coords.shape[0]
    max_adiff = np.max(np.abs(out_dense_np - ref_out_subm_np))
    max_rdiff = max_adiff / np.mean(np.abs(out_dense_np))
    return mean_adiff, max_rdiff


if __name__ == "__main__":
    # Only support single conv layer
    # Cannot support even kernel sizes >= 4 (because of the different definition of anchor point)

    # Set conv_configuration
    config = F.conv_config.get_default_conv_config()
    config.kmap_mode = "hashmap_on_the_fly"
    config.dataflow = F.Dataflow.ImplicitGEMM
    config.ifsort = True
    F.conv_config.set_global_conv_config(config)

    kernel_sizes = [2, 3, 5]
    strides = [1, 2, 3]

    for kernel_size in kernel_sizes:
        config.split_mask_num = kernel_size
        F.conv_config.set_global_conv_config(config)
        for stride in strides:
            mean_adiff, max_rdiff = test_single_layer_convolution_forward(
                kernel_size=kernel_size, stride=stride
            )
            print("****************************")
            print("kernel_size, stride:", kernel_size, stride)
            print("mean_adiff, max_rdiff:", mean_adiff, max_rdiff)
            print("****************************")

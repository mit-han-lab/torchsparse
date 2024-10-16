from typing import Dict, List, Optional, Tuple, Union, Sequence, Mapping, Any
from functools import partial

import torch.nn as nn

from mmdet3d.registry import MODELS
from mmengine.runner import amp

import torchsparse
from torchsparse.nn import functional as F
F.set_conv_mode(1)

import os, logging

from ..layers import SparseBasicBlockTS, make_sparse_convmodule_ts

@MODELS.register_module("SparseEncoderTS")
class SparseEncoderTS(nn.Module):
    r"""Sparse encoder for SECOND and Part-A2.

    Args:
        in_channels (int): The number of input channels.
        sparse_shape (list[int]): The sparse shape of input tensor.
        order (list[str], optional): Order of conv module.
            Defaults to ('conv', 'norm', 'act').
        norm_cfg (dict, optional): Config of normalization layer. Defaults to
            dict(type='BN1d', eps=1e-3, momentum=0.01).
        base_channels (int, optional): Out channels for conv_input layer.
            Defaults to 16.
        output_channels (int, optional): Out channels for conv_out layer.
            Defaults to 128.
        encoder_channels (tuple[tuple[int]], optional):
            Convolutional channels of each encode block.
        encoder_paddings (tuple[tuple[int]], optional):
            Paddings of each encode block.
            Defaults to ((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)).
        block_type (str, optional): Type of the block to use.
            Defaults to 'conv_module'.
    """

    DEFAULT_CONV_CFG = {"type": "TorchSparseConv3d"}
    DEFAULT_NORM_CFG = {"type": "TorchSparseBatchNorm", "eps": 1e-3, "momentum": 0.01}

    def __init__(
        self,
        in_channels: int,
        sparse_shape,
        order: Sequence[str] = ("conv", "norm", "act"),
        norm_cfg: Mapping[str, Any] = DEFAULT_NORM_CFG,
        base_channels: int = 16,
        output_channels: int = 128,
        encoder_channels=((16,), (32, 32, 32), (64, 64, 64), (64, 64, 64)),
        encoder_paddings=((1,), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)),
        block_type: str = "conv_module",
        activation_type: str = "relu",
        return_middle_feats: bool = False
    ) -> None:
        super().__init__()
        assert block_type in ["conv_module", "basicblock"]
        self.sparse_shape = sparse_shape
        self.in_channels = in_channels
        self.order = order
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.encoder_paddings = encoder_paddings
        self.stage_num = len(self.encoder_channels)
        self.fp16_enabled = False
        self.activation_type = activation_type
        self.return_middle_feats = return_middle_feats
        # Spconv init all weight on its own

        assert isinstance(order, (list, tuple)) and len(order) == 3
        assert set(order) == {"conv", "norm", "act"}

        make_block_fn = partial(make_sparse_convmodule_ts, activation_type=activation_type)

        if self.order[0] != "conv":  # pre activate
            self.conv_input = make_block_fn(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                # indice_key="subm1",
                conv_type="TorchSparseConv3d",
                order=("conv",),
            )
        else:  # post activate
            self.conv_input = make_block_fn(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                # indice_key="subm1",
                conv_type="TorchSparseConv3d",
            )

        encoder_out_channels = self.make_encoder_layers(
            make_block_fn, norm_cfg, self.base_channels, block_type=block_type  # make_block_fn is a function to passed in to make blocks
        )

        self.conv_out = make_block_fn(
            encoder_out_channels,
            self.output_channels,
            kernel_size=(3, 1, 1),
            stride=(2, 1, 1),
            norm_cfg=norm_cfg,
            padding=0,
            # indice_key="spconv_down2",
            conv_type="TorchSparseConv3d",
        )
        # print("\033[92m" + "Sparse Encoder" + "\033[0m")
        logging.info("Using TorchSparse SparseEncoder")

    @amp.autocast(enabled=False)
    def forward(self, voxel_features, coors, batch_size, **kwargs):
        """Forward of SparseEncoder.

        Args:
            voxel_features (torch.float32): Voxel features in shape (N, C).
            coors (torch.int32): Coordinates in shape (N, 4),
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size.

        Returns:
            dict: Backbone features.
        """
        coors = coors.int()
        spatial_range = (coors[:, 0].max().item() + 1,) + tuple(self.sparse_shape)
        input_sp_tensor = torchsparse.SparseTensor(voxel_features, coors, spatial_range=spatial_range)
        x = self.conv_input(input_sp_tensor)

        encode_features = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encode_features.append(x)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(encode_features[-1])  # out differs
        spatial_features = out.dense()

        # TS Integrated New sequence
        N, D, H, W, C = spatial_features.shape
        spatial_features = spatial_features.permute(0, 2, 3, 4, 1).contiguous().reshape(N, H, W, C*D).permute(0, 3, 1, 2).contiguous()

        # SpConv Original Sequence
        # N, C, D, H, W = spatial_features.shape
        # spatial_features = spatial_features.view(N, C * D, H, W)

        # N, D, H, W, C = spatial_features.shape
        # spatial_features = spatial_features.reshape(N, H, W, D * C).permute(0, 3, 1, 2).contiguous()
        # # consistent with spconv
        # # spatial_features = spatial_features.transpose(-1, -2).reshape(N, H, W, D * C).permute(0, 3, 1, 2).contiguous()

        if self.return_middle_feats:
            return spatial_features, encode_features
        else:
            return spatial_features

    def make_encoder_layers(
        self,
        make_block,
        norm_cfg,
        in_channels,
        block_type="conv_module",
        conv_cfg=DEFAULT_CONV_CFG,
    ):
        """Make encoder layers using sparse convs.

        Args:
            make_block (method): A bounded function to build blocks.
            norm_cfg (dict[str]): Config of normalization layer.
            in_channels (int): The number of encoder input channels.
            block_type (str, optional): Type of the block to use.
                Defaults to 'conv_module'.
            conv_cfg (dict, optional): Config of conv layer. Defaults to
                dict(type='SubMConv3d').

        Returns:
            int: The number of encoder output channels.
        """
        assert block_type in ["conv_module", "basicblock"]
        self.encoder_layers = nn.Sequential()

        for i, blocks in enumerate(self.encoder_channels):
            blocks_list = []
            for j, out_channels in enumerate(tuple(blocks)):
                padding = tuple(self.encoder_paddings[i])[j]
                # each stage started with a spconv layer
                # except the first stage
                if i != 0 and j == 0 and block_type == "conv_module":
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            stride=2,
                            padding=padding,
                            # indice_key=f"spconv{i + 1}",
                            conv_type="TorchSparseConv3d",
                        )
                    )
                elif block_type == "basicblock":
                    if j == len(blocks) - 1 and i != len(self.encoder_channels) - 1:
                        blocks_list.append(
                            make_block(
                                in_channels,
                                out_channels,
                                3,
                                norm_cfg=norm_cfg,
                                stride=2,
                                padding=padding,
                                # indice_key=f"spconv{i + 1}",
                                conv_type="TorchSparseConv3d",
                            )
                        )
                    else:
                        blocks_list.append(
                            SparseBasicBlockTS(
                                out_channels,
                                out_channels,
                                norm_cfg=norm_cfg,
                                conv_cfg=conv_cfg,
                                act_cfg=self.activation_type,
                            )
                        )
                else:
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            padding=padding,
                            # indice_key=f"subm{i + 1}",
                            conv_type="TorchSparseConv3d",
                        )
                    )
                in_channels = out_channels
            stage_name = f"encoder_layer{i + 1}"
            # stage_layers = spconv.SparseSequential(*blocks_list)
            stage_layers = nn.Sequential(*blocks_list)
            self.encoder_layers.add_module(stage_name, stage_layers)
        return out_channels

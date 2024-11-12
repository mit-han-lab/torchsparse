from torch import nn
import torchsparse.nn as spnn

from ..backbones.resnet import BasicBlockTS
from mmcv.cnn import build_conv_layer, build_norm_layer

import logging

def replace_feature_ts(out,  new_features):
    out.feats = new_features
    return out


class SparseBasicBlockTS(BasicBlockTS):
    """Sparse basic block for PartA^2.

    Sparse basic block implemented with submanifold sparse convolution.

    Args:
        inplanes (int): Inplanes of block.
        planes (int): Planes of block.
        stride (int or Tuple[int]): Stride of the first block. Defaults to 1.
        downsample (Module, optional): Down sample module for block.
            Defaults to None.
        indice_key (str): Indice key for spconv. Default to None.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer. Defaults to None.
    """

    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
    ):
        BasicBlockTS.__init__(
            self,
            inplanes,
            planes,
            stride=stride,
            downsample=downsample,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
        )
        if act_cfg is not None:
            if act_cfg == "swish":
                self.relu = spnn.SiLU(inplace=True)
            else:
                self.relu = spnn.ReLU(inplace=True)



def make_sparse_convmodule_ts(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    conv_type="TorchSparseConv3d",
    norm_cfg=None,
    order=("conv", "norm", "act"),
    activation_type="relu",
    indice_key=None,
    transposed=False
):
    """Make sparse convolution module.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of out channels.
        kernel_size (int | Tuple[int]): Kernel size of convolution.
        indice_key (str): The indice key used for sparse tensor.
        stride (int or tuple[int]): The stride of convolution.
        padding (int or tuple[int]): The padding number of input.
        conv_type (str): Sparse conv type in spconv. Defaults to 'SubMConv3d'.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer. Defaults to None.
        order (Tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
            Defaults to ('conv', 'norm', 'act').

    Returns:
        spconv.SparseSequential: sparse convolution module.
    """
    assert isinstance(order, tuple) and len(order) <= 3
    assert set(order) | {"conv", "norm", "act"} == {"conv", "norm", "act"}

    conv_cfg = {"type": conv_type}

    if norm_cfg is None:
        norm_cfg = dict(type='BN1d')

    layers = []
    for layer in order:
        if layer == "conv":
            layers.append(
                build_conv_layer(
                    cfg=conv_cfg,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                    transposed=transposed,
                )
                # spnn.Conv3d(
                #     in_channels=in_channels,
                #     out_channels=out_channels,
                #     kernel_size=kernel_size,
                #     stride=stride,
                #     padding=padding,
                #     bias=False,
                #     transposed=transposed)
            )
        elif layer == "norm":
            assert norm_cfg is not None, "norm_cfg must be provided"
            layers.append(build_norm_layer(norm_cfg, out_channels)[1])
        elif layer == "act":
            if activation_type == "relu":
                layers.append(spnn.ReLU(inplace=True))
            elif activation_type == "swish":
               layers.append(spnn.SiLU(inplace=True))
            else:
                raise NotImplementedError
    layers = nn.Sequential(*layers)
    logging.info("Made TorchSparse Module")
    return layers
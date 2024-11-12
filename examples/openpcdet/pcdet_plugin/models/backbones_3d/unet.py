from functools import partial

import torch
import torch.nn as nn

from pcdet.utils.spconv_utils import spconv
from pcdet.utils import common_utils
from .backbone3d import post_act_block_ts

import os

import torchsparse.nn as spnn
import torchsparse

def ts_replace_feature(x: torchsparse.SparseTensor, features):
    x.feats = features
    return x

class SparseBasicBlockTS(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, indice_key=None, norm_fn=None):
        super(SparseBasicBlockTS, self).__init__()
        self.conv1 = spnn.Conv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = norm_fn(planes)
        self.relu = spnn.ReLU()
        self.conv2 = spnn.Conv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torchsparse.SparseTensor):
        identity = x.feats

        assert x.feats.dim() == 2, 'x.features.dim()=%d' % x.features.dim()

        out: torchsparse.SparseTensor = self.conv1(x)
        # out = ts_replace_feature(out, self.bn1(out.feats))
        # out = ts_replace_feature(out, self.relu(out.feats))
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = ts_replace_feature(out, self.bn2(out.feats))
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = ts_replace_feature(out, out.feats + identity)
        # out = ts_replace_feature(out, self.relu(out.feats))
        out = self.relu(out)
        return out


class UNetV2TS(nn.Module):
    """
    Sparse Convolution based UNet for point-wise feature learning.
    Reference Paper: https://arxiv.org/abs/1907.03670 (Shaoshuai Shi, et. al)
    From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    """

    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        norm_fn = partial(spnn.BatchNorm, eps=1e-3, momentum=0.01)

        self.conv_input = nn.Sequential(
            spnn.Conv3d(input_channels, 16, 3, padding=1, bias=False),
            norm_fn(16),
            spnn.ReLU(),
        )
        block = post_act_block_ts

        self.conv1 = nn.Sequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1),
        )

        self.conv2 = nn.Sequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1), 
            block(32, 32, 3, norm_fn=norm_fn, padding=1),
            block(32, 32, 3, norm_fn=norm_fn, padding=1),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3'),  # Notice that  conv_type='spconv' is replaced
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1)),
            block(64, 64, 3, norm_fn=norm_fn, padding=1),
            block(64, 64, 3, norm_fn=norm_fn, padding=1),
        )

        if self.model_cfg.get('RETURN_ENCODED_TENSOR', True):
            last_pad = self.model_cfg.get('last_pad', 0)

            self.conv_out = nn.Sequential(
                # [200, 150, 5] -> [200, 150, 2]
                spnn.Conv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                    bias=False),
                norm_fn(128),
                spnn.ReLU(),
            )
        else:
            self.conv_out = None

        # decoder
        # [400, 352, 11] <- [200, 176, 5]
        self.conv_up_t4 = SparseBasicBlockTS(64, 64, indice_key='subm4', norm_fn=norm_fn)
        self.conv_up_m4 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
        self.inv_conv4 = block(64, 64, 3, stride=2, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')

        # [800, 704, 21] <- [400, 352, 11]
        self.conv_up_t3 = SparseBasicBlockTS(64, 64, indice_key='subm3', norm_fn=norm_fn)
        self.conv_up_m3 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3')
        self.inv_conv3 = block(64, 32, 3, stride=2, norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')

        # [1600, 1408, 41] <- [800, 704, 21]
        self.conv_up_t2 = SparseBasicBlockTS(32, 32, indice_key='subm2', norm_fn=norm_fn)
        self.conv_up_m2 = block(64, 32, 3, norm_fn=norm_fn, indice_key='subm2')
        self.inv_conv2 = block(32, 16, 3, stride=2, norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv')

        # [1600, 1408, 41] <- [1600, 1408, 41]
        self.conv_up_t1 = SparseBasicBlockTS(16, 16, indice_key='subm1', norm_fn=norm_fn)
        self.conv_up_m1 = block(32, 16, 3, norm_fn=norm_fn, indice_key='subm1')

        self.conv5 = nn.Sequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1')
        )
        self.num_point_features = 16

    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
        x_trans = conv_t(x_lateral)
        x = x_trans
        x = ts_replace_feature(x, torch.cat((x_bottom.feats, x_trans.feats), dim=1))  # was dim 0
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.feats.shape[1])
        x = ts_replace_feature(x, x_m.feats + x.feats)
        x = conv_inv(x)
        return x

    @staticmethod
    def channel_reduction(x: torchsparse.SparseTensor, out_channels):
        """
        Args:
            x: x.features (N, C1)
            out_channels: C2

        Returns:

        """
        features = x.feats
        n, in_channels = features.shape
        assert (in_channels % out_channels == 0) and (in_channels >= out_channels)

        x = ts_replace_feature(x, features.view(n, out_channels, -1).sum(dim=2))
        return x

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = torchsparse.SparseTensor(
            feats=voxel_features,
            coords=voxel_coords.int(),
            spatial_range=(batch_size,) + tuple(self.sparse_shape)
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        if self.conv_out is not None:
            # for detection head
            # [200, 176, 5] -> [200, 176, 2]
            out = self.conv_out(x_conv4)
            batch_dict['encoded_spconv_tensor'] = out
            batch_dict['encoded_spconv_tensor_stride'] = 8

        # for segmentation head
        # [400, 352, 11] <- [200, 176, 5]
        x_up4 = self.UR_block_forward(x_conv4, x_conv4, self.conv_up_t4, self.conv_up_m4, self.inv_conv4)
        # [800, 704, 21] <- [400, 352, 11]
        x_up3 = self.UR_block_forward(x_conv3, x_up4, self.conv_up_t3, self.conv_up_m3, self.inv_conv3)
        # [1600, 1408, 41] <- [800, 704, 21]
        x_up2 = self.UR_block_forward(x_conv2, x_up3, self.conv_up_t2, self.conv_up_m2, self.inv_conv2)
        # [1600, 1408, 41] <- [1600, 1408, 41]
        x_up1 = self.UR_block_forward(x_conv1, x_up2, self.conv_up_t1, self.conv_up_m1, self.conv5)

        batch_dict['point_features'] = x_up1.feats
        point_coords = common_utils.get_voxel_centers(
            x_up1.coords[:, 1:], downsample_times=1, voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )
        batch_dict['point_coords'] = torch.cat((x_up1.coords[:, 0:1].float(), point_coords), dim=1)
        return batch_dict

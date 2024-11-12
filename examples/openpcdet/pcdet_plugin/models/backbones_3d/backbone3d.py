from functools import partial

import torch
import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn

import os, logging

from pcdet.models.backbones_3d.__init__ import __all__

def post_act_block_ts(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='tsconv', norm_fn=None):

    if conv_type == 'tsconv':
        conv = spnn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
    elif conv_type == 'inverseconv':
        conv = spnn.Conv3d(in_channels, out_channels, kernel_size,  stride=stride, bias=False, transposed=True) 
    else:
        raise NotImplementedError

    m = nn.Sequential(
        conv,
        norm_fn(out_channels),
        spnn.ReLU(),
    )

    return m


class VoxelBackBone8xTS(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(spnn.BatchNorm, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = nn.Sequential(
            spnn.Conv3d(input_channels, 16, 3, padding=1, bias=False),
            spnn.BatchNorm(16),
            spnn.ReLU(),
        )

        block = post_act_block_ts

        self.conv1 = nn.Sequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = nn.Sequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='tsconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = nn.Sequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='tsconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = nn.Sequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='tsconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = nn.Sequential(
            # [200, 150, 5] -> [200, 150, 2]
            spnn.Conv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False),
            norm_fn(128),
            spnn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }

        logging.warning('Built VoxelBackBone8x for TorchSparse')


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
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        # input_sp_tensor = spconv.SparseConvTensor(
        #     features=voxel_features,
        #     indices=voxel_coords.int(),
        #     spatial_shape=self.sparse_shape,
        #     batch_size=batch_size
        # )
        voxel_coords = voxel_coords.int()
        # input_sp_tensor = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        spatial_range = (voxel_coords[:, 0].max().item() + 1,) + tuple(self.sparse_shape)
        input_sp_tensor = torchsparse.SparseTensor(voxel_features, voxel_coords, spatial_range=spatial_range)  # dimension match
        

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict

__all__['VoxelBackBone8xTS'] = VoxelBackBone8xTS
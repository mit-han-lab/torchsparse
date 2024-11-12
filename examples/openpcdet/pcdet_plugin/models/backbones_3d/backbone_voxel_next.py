from functools import partial
import torch
import torch.nn as nn
import os
from pcdet.utils.spconv_utils import spconv
import torchsparse
import torchsparse.nn as spnn
from torchsparse.utils.tensor_cache import TensorCache
import logging


def ts_to_spconv(tensor: torchsparse.SparseTensor, spatial_shape, batch_size):
    return spconv.SparseConvTensor(
        features=tensor.feats,
        indices=tensor.coords,
        spatial_shape=spatial_shape,  # Take out batch size and channel
        batch_size=batch_size
    )

def spconv_to_ts(tensor: spconv.SparseConvTensor):
    spatial_range = None
    if len(tensor.indices[0]) == 3:
        spatial_range = (tensor.batch_size, ) + tuple(tensor.spatial_shape) + (1, )
    elif len(tensor.indices[0]) == 4:
        spatial_range = (tensor.batch_size, ) + (tensor.spatial_shape, )
    else:
        raise NotImplementedError("Only 3D and 4D tensors are supported.")

    return torchsparse.SparseTensor(
        feats=tensor.features,
        coords=tensor.indices,
        spatial_range=spatial_range
    )
    

def post_act_block_ts(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spnn.Conv3d(in_channels, out_channels, kernel_size, bias=False)
    elif conv_type == 'spconv':
        conv = spnn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
    elif conv_type == 'inverseconv':
        conv = spnn.Conv3d(in_channels, out_channels, kernel_size, bias=False, transposed=True)
    else:
        raise NotImplementedError

    m = nn.Sequential(
        conv,
        norm_fn(out_channels),
        spnn.ReLU(),
    )

    return m


class SparseBasicBlockTS(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlockTS, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spnn.Conv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias
        )
        self.bn1 = nn.BatchNorm1d(planes, eps=1e-3, momentum=0.01)  # norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spnn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias
        )
        self.bn2 = nn.BatchNorm1d(planes, eps=1e-3, momentum=0.01)  # norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out.feats = self.bn1(out.feats)
        out.feats = self.relu(out.feats)

        out = self.conv2(out)
        out.feats = self.bn2(out.feats)

        if self.downsample is not None:
            identity = self.downsample(x)

        out.feats = out.feats + identity.feats
        out.feats = self.relu(out.feats)

        return out

class VoxelResBackBone8xVoxelNeXtTS(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
#        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        norm_fn = partial(torchsparse.nn.BatchNorm, eps=1e-3, momentum=0.01)

        spconv_kernel_sizes = model_cfg.get('SPCONV_KERNEL_SIZES', [3, 3, 3, 3])
        channels = model_cfg.get('CHANNELS', [16, 32, 64, 128, 128])
        out_channel = model_cfg.get('OUT_CHANNEL', 128)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = nn.Sequential(
            spnn.Conv3d(input_channels, channels[0], 3, padding=1, bias=False),
            norm_fn(channels[0]),
            spnn.ReLU()
        )
        block = post_act_block_ts

        self.conv1 = nn.Sequential(
            SparseBasicBlockTS(channels[0], channels[0], norm_fn=norm_fn),
            SparseBasicBlockTS(channels[0], channels[0], norm_fn=norm_fn),
        )

        self.conv2 = nn.Sequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(channels[0], channels[1], spconv_kernel_sizes[0], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[0]//2), conv_type='spconv'),
            SparseBasicBlockTS(channels[1], channels[1], norm_fn=norm_fn),
            SparseBasicBlockTS(channels[1], channels[1], norm_fn=norm_fn),
        )

        self.conv3 = nn.Sequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(channels[1], channels[2], spconv_kernel_sizes[1], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[1]//2), conv_type='spconv'),
            SparseBasicBlockTS(channels[2], channels[2], norm_fn=norm_fn),
            SparseBasicBlockTS(channels[2], channels[2], norm_fn=norm_fn),
        )

        self.conv4 = nn.Sequential(
            # [400, 352, 11] <- [200, 176, 6]
            block(channels[2], channels[3], spconv_kernel_sizes[2], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[2]//2), conv_type='spconv'),
            SparseBasicBlockTS(channels[3], channels[3], norm_fn=norm_fn),
            SparseBasicBlockTS(channels[3], channels[3], norm_fn=norm_fn),
        )

        self.conv5 = nn.Sequential(
            # [200, 176, 6] <- [100, 88, 3]
            block(channels[3], channels[4], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3]//2), conv_type='spconv'),
            SparseBasicBlockTS(channels[4], channels[4], norm_fn=norm_fn),
            SparseBasicBlockTS(channels[4], channels[4], norm_fn=norm_fn),
        )
        
        self.conv6 = nn.Sequential(
            # [200, 176, 6] <- [100, 88, 3]
            block(channels[4], channels[4], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3]//2), conv_type='spconv'),
            SparseBasicBlockTS(channels[4], channels[4], norm_fn=norm_fn),
            SparseBasicBlockTS(channels[4], channels[4], norm_fn=norm_fn),
        )
        # self.conv_out = nn.Sequential(
        #     # [200, 150, 5] -> [200, 150, 2], zyx in SpConv -> zyx in TS
        #     spnn.Conv3d(channels[3], out_channel, (1,3,3), padding=1, bias=False),
        #     norm_fn(out_channel),
        #     spnn.ReLU(),
        # )

        self.conv_out = spconv.SparseSequential(
            spconv.SparseConv2d(channels[3], out_channel, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_channel,eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        self.shared_conv = nn.Sequential(
            spnn.Conv3d(out_channel, out_channel, kernel_size=(3,3,1), bias=True),
            spnn.BatchNorm(out_channel),
            spnn.ReLU(),
        )

        self.forward_ret_dict = {}
        self.num_point_features = out_channel
        self.backbone_channels = {
            'x_conv1': channels[0],
            'x_conv2': channels[1],
            'x_conv3': channels[2],
            'x_conv4': channels[3]
        }
        logging.info('VoxelNeXt TorchSparse')

    def bev_out(self, x_conv):
        features_cat = x_conv.feats
        indices_cat = x_conv.coords[:, [0, 2, 3]]
        spatial_shape = x_conv.spatial_range[1:-1]
        channels = x_conv.spatial_range[-1]

        indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True)
        features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1]))
        features_unique.index_add_(0, _inv, features_cat)

        x_out = torchsparse.SparseTensor(
            feats=features_unique,
            coords=indices_unique,
            spatial_range=(x_conv.spatial_range[0],) + spatial_shape + (channels, )
        )
        return x_out

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
        input_sp_tensor = torchsparse.SparseTensor(
            feats=voxel_features,
            coords=voxel_coords.int(),
            spatial_range=(batch_size, ) + tuple(self.sparse_shape) + (4, )
        )
        # input_sp_tensor = spconv.SparseConvTensor(
        #     features=voxel_features,
        #     indices=voxel_coords.int(),
        #     spatial_shape=self.sparse_shape,
        #     batch_size=batch_size
        # )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv5 = self.conv5(x_conv4)
        x_conv6 = self.conv6(x_conv5)

        # print("x_conv6.feats")
        # print(x_conv6.feats)

        x_conv5.coords[:, 1:] *= 2
        x_conv6.coords[:, 1:] *= 4
        x_conv4.feats = torch.cat([x_conv4.feats, x_conv5.feats, x_conv6.feats])
        x_conv4.coords = torch.cat([x_conv4.coords, x_conv5.coords, x_conv6.coords])


        out = self.bev_out(x_conv4)

        # out = ts_to_spconv(out)
        # out.spatal_range = out.spatial_range + (1, )

        out = ts_to_spconv(out, spatial_shape=out.spatial_range[2:], batch_size=out.spatial_range[0])
        out = self.conv_out(out)
        out = spconv_to_ts(out)
        out.coords = torch.cat((out.coords, torch.zeros((out.coords.shape[0], 1)).to('cuda')), dim=1).int()
        out = self.shared_conv(out)

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
        # print("out.feats")
        # print(out.feats)

        # print("batch_dict")
        # print(batch_dict)

        return batch_dict

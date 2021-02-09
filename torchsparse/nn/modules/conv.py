import math

import torch
from torch import nn

from torchsparse.sparse_tensor import *

from ..functional import *

__all__ = ['Conv3d', 'ToBEVConvolution', 'ToBEVReduction', 'ToDenseBEVConvolution']


class Conv3d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 dilation: int = 1,
                 bias: bool = False,
                 transpose: bool = False) -> None:
        super().__init__()
        self.in_channels = inc = in_channels
        self.out_channels = outc = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        if not isinstance(kernel_size, (list, tuple)):
            self.kernel_volume = self.kernel_size ** 3
            self.kernel = nn.Parameter(
                torch.zeros(self.kernel_volume, inc,
                            outc)) if self.kernel_size > 1 else nn.Parameter(
                                torch.zeros(inc, outc))
        else:
            if len(self.kernel_size) == 3:
                self.kernel_volume = self.kernel_size[0]*self.kernel_size[1]*self.kernel_size[2]
                self.kernel = nn.Parameter(torch.zeros(self.kernel_volume, inc, outc))
            else:
                raise ValueError("kernel_size must be either an integer of a 3 dimensional tuple")


        self.bias = None if not bias else nn.Parameter(torch.zeros(outc))
        self.t = transpose
        self.init_weight()

        if kernel_size == 1:
            assert not transpose

    def __repr__(self):
        if not self.t:
            return 'Conv3d(in_channels=%d, out_channels=%d, kernel_size=%d, stride=%d, dilation=%d)' % (
                self.in_channels, self.out_channels, self.kernel_size,
                self.stride, self.dilation)
        else:
            return 'Conv3d(in_channels=%d, out_channels=%d, kernel_size=%d, stride=%d, dilation=%d)' % (
                self.in_channels, self.out_channels, self.kernel_size,
                self.stride, self.dilation)

    def init_weight(self):
        std = 1. / math.sqrt(
            self.out_channels if self.t else self.in_channels *
            (self.kernel_volume))
        self.kernel.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, inputs):
        return conv3d(inputs,
                      self.kernel,
                      ks=self.kernel_size,
                      bias=self.bias,
                      stride=self.stride,
                      dilation=self.dilation,
                      transpose=self.t)

class ToBEVConvolution(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 n_kernels: int, 
                 stride: int = 1, 
                 z_dim: int = 1, 
                 use_bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.z_dim = z_dim
        self.kernel = nn.Parameter(torch.zeros(n_kernels, in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(1, out_channels)) if use_bias else 0
        self.init_weight()

    def init_weight(self):
        std = 1. / math.sqrt(self.in_channels)
        self.kernel.data.uniform_(-std, std)
        
    def __repr__(self):
        return 'ToBEVConvolution(in_channels=%d, out_channels=%d, n_kernels=%d, stride=%d)'%(
            self.in_channels,
            self.out_channels,
            self.n_kernels,
            self.stride
        )

    def forward(self, inputs):
        features = inputs.F
        coords = inputs.C
        cur_stride = inputs.s
        ratio = cur_stride * self.stride

        kernels = torch.index_select(self.kernel, 0, coords[:, self.z_dim].long() / cur_stride)
        output_features = (features.unsqueeze(-1) * kernels).sum(1) + self.bias
        output_coords = coords.t().long()
        output_coords[self.z_dim, :] = 0
        if self.stride > 1:
            output_coords[:3] /= ratio
            output_coords[:3] *= ratio
        flatten = torch.cuda.sparse.FloatTensor(output_coords, output_features).coalesce()
        return SparseTensor(flatten.values(), flatten.indices().t().int(), ratio)


class ToBEVReduction(nn.Module):
    def __init__(self, 
                 z_dim: int = 1):
        super().__init__()
        self.z_dim = z_dim
        
    def __repr__(self):
        return 'ToBEVReduction(z_dim = %d)'%self.z_dim

    def forward(self, inputs):
        features = inputs.F
        coords = inputs.C
        cur_stride = inputs.s

        flatten_coords = coords.clone()
        flatten_coords[:, self.z_dim] = 0
        features_with_cnt = torch.cat([torch.ones_like(features[:, :1]), features], axis=1)
        flatten = torch.cuda.sparse.FloatTensor(flatten_coords.t().long(), features_with_cnt).coalesce()
        output_features = flatten.values()[:, 1:] / flatten.values()[:, :1]
        return SparseTensor(output_features, flatten.indices().t().int(), cur_stride)


class ToDenseBEVConvolution(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 shape, 
                 offset: list = [0,0,0], 
                 z_dim: int = 1, 
                 use_bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.offset = torch.cuda.IntTensor([list(offset) + [0]])
        self.z_dim = z_dim
        self.n_kernels = int(shape[self.z_dim])
        self.bev_dims = [i for i in range(3) if i != self.z_dim]
        self.bev_shape = shape[self.bev_dims]
        self.kernel = nn.Parameter(torch.zeros(self.n_kernels, in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(1, out_channels)) if use_bias else 0
        self.init_weight()
        
    def __repr__(self):
        return 'ToDenseBEVConvolution(in_channels=%d, out_channels=%d, n_kernels=%d)'%(
            self.in_channels,
            self.out_channels,
            self.n_kernels
        )


    def init_weight(self):
        std = 1. / math.sqrt(self.in_channels)
        self.kernel.data.uniform_(-std, std)

    def forward(self, inputs):
        features = inputs.F
        coords = inputs.C
        cur_stride = inputs.s

        kernels = torch.index_select(self.kernel, 0, coords[:, self.z_dim].long() / cur_stride)
        sparse_features = (features.unsqueeze(-1) * kernels).sum(1) + self.bias
        sparse_coords = (coords - self.offset).t()[[3] + self.bev_dims].long()
        sparse_coords[1:] /= cur_stride
        batch_size = sparse_coords[0].max().item() + 1
        sparse_coords = sparse_coords[0] * int(self.bev_shape.prod()) + sparse_coords[1] * int(self.bev_shape[1]) + sparse_coords[2]
        bev = torch.cuda.sparse.FloatTensor(
            sparse_coords.unsqueeze(0),
            sparse_features,
            torch.Size([batch_size * int(self.bev_shape.prod()), sparse_features.size(-1)]),
        ).to_dense()
        return bev.view(batch_size, *self.bev_shape, -1).permute(0, 3, 1, 2).contiguous() # To BCHW

import math

import torch
from torch import nn
from torchsparse.sparse_tensor import *

from ..functional import *

from typing import Union, List, Tuple

__all__ = [
    'Conv3d', 'ToBEVConvolution', 'ToBEVReduction', 'ToDenseBEVConvolution',
    'ToBEVHeightCompression'
]


class Conv3d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, int, int]] = 3,
                 stride: Union[int, List[int], Tuple[int, int, int]] = 1,
                 dilation: int = 1,
                 bias: bool = False,
                 transpose: bool = False) -> None:
        super().__init__()
        self.in_channels = inc = in_channels
        self.out_channels = outc = out_channels
        if isinstance(kernel_size, list):
            self.kernel_size = tuple(kernel_size)
        else:
            self.kernel_size = kernel_size
        if isinstance(stride, list):
            self.stride = tuple(stride)
        else:
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
                self.kernel_volume = self.kernel_size[0] * self.kernel_size[
                    1] * self.kernel_size[2]
                self.kernel = nn.Parameter(
                    torch.zeros(self.kernel_volume, inc, outc))
            else:
                raise ValueError(
                    "kernel_size must be either an integer of a 3 dimensional tuple"
                )

        self.bias = None if not bias else nn.Parameter(torch.zeros(outc))
        self.t = transpose
        self.reset_parameters()

        if kernel_size == 1:
            assert not transpose

    def __repr__(self):
        if not self.t:
            return 'Conv3d(in_channels={}, out_channels={}, kernel_size={}, stride={}, dilation={})'.format(
                self.in_channels, self.out_channels, self.kernel_size, self.stride, self.dilation)
        else:
            return 'Conv3dTranspose(in_channels={}, out_channels={}, kernel_size={}, stride={}, dilation={})'.format(
                self.in_channels, self.out_channels, self.kernel_size, self.stride, self.dilation)

    def reset_parameters(self):
        std = 1. / math.sqrt(
            self.out_channels if self.t else self.in_channels *
            (self.kernel_volume))
        self.kernel.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, inputs: SparseTensor) -> SparseTensor:
        return conv3d(inputs,
                      self.kernel,
                      kernel_size=self.kernel_size,
                      bias=self.bias,
                      stride=self.stride,
                      dilation=self.dilation,
                      transpose=self.t)


class ToBEVReduction(nn.Module):
    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def extra_repr(self):
        return 'dim = {}'.format(self.dim)

    def forward(self, inputs: SparseTensor) -> SparseTensor:
        coords, feats, stride = inputs.C, inputs.F, inputs.s

        coords = coords.clone()
        coords[:, self.dim] = 0
        feats = torch.cat([torch.ones_like(feats[:, :1]), feats], axis=1)
        tensor = torch.cuda.sparse.FloatTensor(coords.t().long(),
                                               feats).coalesce()
        coords = tensor.indices().t().int()
        feats = tensor.values()[:, 1:] / tensor.values()[:, :1]
        return SparseTensor(coords=coords, feats=feats, stride=stride)


class ToBEVConvolution(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 n_kernels: int,
                 stride: int = 1,
                 dim: int = 1,
                 bias: bool = False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_kernels = n_kernels
        self.stride = stride
        self.dim = dim
        self.kernel = nn.Parameter(
            torch.zeros(n_kernels, in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(1, out_channels)) if bias else 0
        self.reset_parameters()

    def reset_parameters(self):
        std = 1. / math.sqrt(self.in_channels)
        self.kernel.data.uniform_(-std, std)

    def extra_repr(self):
        return 'in_channels={}, out_channels={}, n_kernels={}, stride={}'.format(
            self.in_channels, self.out_channels, self.n_kernels, self.stride)

    def forward(self, inputs: SparseTensor) -> SparseTensor:
        coords, feats, stride = inputs.C, inputs.F, inputs.s
        ratio = stride * self.stride
        if isinstance(stride, tuple):
            stride = torch.Tensor(stride).unsqueeze(0).to(feats)[
                :, self.dim
            ]

        kernels = torch.index_select(self.kernel, 0,
                                     coords[:, self.dim].long() // stride)
        feats = (feats.unsqueeze(-1) * kernels).sum(1) + self.bias
        coords = coords.t().long()
        coords[self.dim, :] = 0
        if self.stride > 1:
            coords[:3] /= ratio
            coords[:3] *= ratio
        flatten = torch.cuda.sparse.FloatTensor(coords, feats).coalesce()
        return SparseTensor(flatten.values(),
                            flatten.indices().t().int(), ratio)


class ToDenseBEVConvolution(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 shape,
                 offset: List[int] = [0, 0, 0],
                 dim: int = 1,
                 bias: bool = False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.register_buffer('offset', torch.IntTensor([list(offset) + [0]]))
        self.dim = dim
        self.n_kernels = int(shape[self.dim])
        self.bev_dims = [i for i in range(3) if i != self.dim]
        self.bev_shape = shape[self.bev_dims]
        self.kernel = nn.Parameter(
            torch.zeros(self.n_kernels, in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(1, out_channels)) if bias else 0
        self.reset_parameters()

    def extra_repr(self):
        return 'in_channels={}, out_channels={}, n_kernels={}'.format(
            self.in_channels, self.out_channels, self.n_kernels)

    def reset_parameters(self):
        std = 1. / math.sqrt(self.in_channels)
        self.kernel.data.uniform_(-std, std)

    def forward(self, inputs: SparseTensor) -> SparseTensor:
        coords, feats, stride = inputs.C, inputs.F, inputs.s
        if isinstance(stride, tuple):
            stride = torch.Tensor(stride).unsqueeze(0).to(feats)[
                :, self.dim
            ]

        kernel = torch.index_select(self.kernel, 0,
                                    (coords[:, self.dim] // stride).long())
        feats = (feats.unsqueeze(-1) * kernel).sum(1) + self.bias
        coords = (coords - self.offset).t()[[3] + self.bev_dims].long()
        coords[1:] = (coords[1:] // stride).long()
        indices = coords[0] * int(self.bev_shape.prod()) + coords[1] * int(
            self.bev_shape[1]) + coords[2]
        batch_size = coords[0].max().item() + 1
        outputs = torch.sparse_coo_tensor(
            indices.unsqueeze(0),
            feats,
            torch.Size(
                [batch_size * int(self.bev_shape.prod()),
                 feats.size(-1)]),
        ).to_dense()
        outputs = outputs.view(batch_size, *self.bev_shape, -1)
        outputs = outputs.permute(0, 3, 1, 2).contiguous()
        return outputs


class ToBEVHeightCompression(nn.Module):
    def __init__(self,
                 channels: int,
                 shape,
                 offset: List[int] = [0, 0, 0],
                 dim: int = 1,
                 bias: bool = False) -> None:
        super().__init__()
        self.channels = channels
        self.register_buffer('offset', torch.IntTensor([list(offset) + [0]]))
        self.dim = dim
        self.bev_dims = [i for i in range(3) if i != self.dim]
        self.bev_shape = shape[self.bev_dims].int()
        self.shape = shape.int()

    def extra_repr(self):
        return 'channels={}'.format(self.channels)

    def forward(self, inputs: SparseTensor) -> SparseTensor:
        coords, feats, stride = inputs.C, inputs.F, inputs.s
        if isinstance(stride, tuple):
            stride = torch.Tensor(stride).unsqueeze(0).to(feats)
        
        # [b, x, y, z]
        coords = (coords - self.offset).t()[[3] + self.bev_dims +
                                            [self.dim]].long()
        shape = self.shape[self.bev_dims + [self.dim]]
        if not isinstance(stride, int):
            dim = self.dim
            stride = stride[:, self.bev_dims + [self.dim]]
            stride = stride.t()
        coords[1:] = (coords[1:] // stride).long()
        coords[-1] = torch.clamp(coords[-1], 0, shape[-1] - 1)
        indices = coords[0] * int(shape.prod()) + coords[1] * int(
            shape[1:].prod()) + coords[2] * int(shape[2]) + coords[3]
        batch_size = coords[0].max().item() + 1
        outputs = torch.sparse_coo_tensor(
            indices.unsqueeze(0),
            feats,
            torch.Size([batch_size * int(self.shape.prod()),
                        feats.size(-1)]),
        ).to_dense()
        outputs = outputs.view(batch_size, *self.bev_shape.cpu().numpy(), -1)
        outputs = outputs.permute(0, 3, 1, 2).contiguous()
        return outputs

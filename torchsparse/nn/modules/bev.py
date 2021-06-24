import math
from typing import List, Tuple, Union

import torch
from torch import nn

from torchsparse import SparseTensor

__all__ = [
    'ToBEVConvolution', 'ToBEVReduction', 'ToDenseBEVConvolution',
    'ToBEVHeightCompression'
]


class ToBEVReduction(nn.Module):

    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def extra_repr(self):
        return f'dim = {self.dim}'

    def forward(self, input: SparseTensor) -> SparseTensor:
        coords, feats, stride = input.C, input.F, input.s

        coords = coords.clone()
        coords[:, self.dim] = 0
        feats = torch.cat([torch.ones_like(feats[:, :1]), feats], axis=1)
        tensor = torch.cuda.sparse.FloatTensor(coords.t().long(),
                                               feats).coalesce()
        coords = tensor.indices().t().int()
        feats = tensor.values()[:, 1:] / tensor.values()[:, :1]
        return SparseTensor(coords=coords, feats=feats, stride=stride)


class ToDenseBEVConvolution(nn.Module):
    """

    Converts a torchsparse.SparseTensor to a BEV feature map.
    Group points with the same z value together and apply the same FC kernel.
    Aggregate the results by summing up all features within one BEV grid.

    in_channels: input channels
    out_channels: output channels
    shape: shape of BEV map.
    dim: dimension index for z. (default: 1 for KITTI coords)
    bias: whether to use bias.

    Warning: usually larger memory consumption than ToBEVHeightCompression.


    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 shape: Union[List[int], Tuple[int, int, int], torch.Tensor],
                 offset: Tuple[int, int, int] = (0, 0, 0),
                 dim: int = 1,
                 bias: bool = False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.register_buffer('offset', torch.IntTensor([list(offset) + [0]]))
        if isinstance(shape, torch.Tensor):
            self.register_buffer('shape', shape.int())
        else:
            self.register_buffer('shape', torch.IntTensor(shape))
        self.dim = dim
        self.n_kernels = int(self.shape[self.dim])
        self.bev_dims = [i for i in range(3) if i != self.dim]
        self.bev_shape = self.shape[self.bev_dims]
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

    def forward(self, input: SparseTensor) -> torch.Tensor:
        coords, feats, stride = input.C, input.F, input.s
        if isinstance(stride, tuple):
            stride = torch.Tensor(stride).unsqueeze(0).to(feats)[:, self.dim]

        kernel = torch.index_select(self.kernel, 0,
                                    (coords[:, self.dim] // stride).long())
        feats = (feats.unsqueeze(-1) * kernel).sum(1) + self.bias
        coords = (coords - self.offset).t()[[3] + self.bev_dims].long()
        coords[1:] = (coords[1:] // stride).long()
        indices = coords[0] * int(self.bev_shape.prod()) + coords[1] * int(
            self.bev_shape[1]) + coords[2]
        batch_size = coords[0].max().item() + 1
        output = torch.sparse_coo_tensor(
            indices.unsqueeze(0),
            feats,
            torch.Size(
                [batch_size * int(self.bev_shape.prod()),
                 feats.size(-1)]),
        ).to_dense()
        output = output.view(batch_size, *self.bev_shape, -1)
        output = output.permute(0, 3, 1, 2).contiguous()
        return output


class ToBEVConvolution(nn.Module):
    """ Sparse version of ToDenseBEVConvolution.
    """

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

    def forward(self, input: SparseTensor) -> torch.Tensor:
        coords, feats, stride = input.C, input.F, input.s
        ratio = stride * self.stride
        if isinstance(stride, tuple):
            stride = torch.Tensor(stride).unsqueeze(0).to(feats)[:, self.dim]

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


class ToBEVHeightCompression(nn.Module):
    """

    Converts a torchsparse.SparseTensor to a dense volumetric tensor,
    then flatten the z dimension.
    E.g. input [N, C] (assume batch_size=1), spatial size [128,2,128]
    then output will be [1, 2C, 128, 128]

    channels: input channels
    (Note: output channels = channels x #unique z values)
    shape: shape of BEV map.
    dim: dimension index for z. (default: 1 for KITTI coords)
    bias: whether to use bias.


    """

    def __init__(self,
                 channels: int,
                 shape: Union[List[int], Tuple[int, int, int], torch.Tensor],
                 offset: Tuple[int, int, int] = (0, 0, 0),
                 dim: int = 1,
                 bias: bool = False) -> None:
        super().__init__()
        self.channels = channels
        self.register_buffer('offset', torch.IntTensor([list(offset) + [0]]))
        if isinstance(shape, torch.Tensor):
            self.register_buffer('shape', shape.int())
        else:
            self.register_buffer('shape', torch.IntTensor(shape))
        self.dim = dim
        self.bev_dims = [i for i in range(3) if i != self.dim]
        self.bev_shape = self.shape[self.bev_dims]

    def extra_repr(self) -> str:
        return f'channels={self.channels}'

    def forward(self, input: SparseTensor) -> torch.Tensor:
        coords, feats, stride = input.C, input.F, input.s
        if isinstance(stride, tuple):
            stride = torch.Tensor(stride).unsqueeze(0).to(feats)
        assert isinstance(stride, torch.Tensor)

        # [b, x, y, z]
        coords = (coords - self.offset).t()[[3] + self.bev_dims
                                            + [self.dim]].long()
        shape = self.shape[self.bev_dims + [self.dim]]

        # now stride must be torch.Tensor since input.s is tuple.
        stride = stride[:, self.bev_dims + [self.dim]].t()

        coords[1:] = (coords[1:] // stride).long()
        coords[-1] = torch.clamp(coords[-1], 0, shape[-1] - 1)
        indices = coords[0] * int(shape.prod()) + coords[1] * int(
            shape[1:].prod()) + coords[2] * int(shape[2]) + coords[3]
        batch_size = coords[0].max().item() + 1
        output = torch.sparse_coo_tensor(
            indices.unsqueeze(0),
            feats,
            torch.Size([batch_size * int(self.shape.prod()),
                        feats.size(-1)]),
        ).to_dense()
        output = output.view(batch_size, *self.bev_shape.cpu().numpy(), -1)
        output = output.permute(0, 3, 1, 2).contiguous()
        return output

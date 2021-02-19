import copy
from typing import Optional

import torch
import torchsparse_backend
from torch.autograd import Function
from torchsparse import *
from torchsparse.utils.kernel_region import *

from .convert_neighbor_map import *
from .downsample import *
from .hash import *

__all__ = ['conv3d']


class SpConvolution(Function):
    @staticmethod
    def forward(ctx,
                inputs,
                kernel,
                neighbor_map,
                neighbor_offset,
                sizes,
                transpose=False):
        inputs = inputs.contiguous()
        kernel = kernel.contiguous()
        if not transpose:
            out = torch.zeros(sizes[1], kernel.size(-1), device=inputs.device)
        else:
            # tbd: ensure the original, upsampled size to be the same.
            out = torch.zeros(sizes[0], kernel.size(-1), device=inputs.device)

        if 'cuda' in str(inputs.device):
            torchsparse_backend.sparseconv_forward(inputs, out, kernel,
                                                   neighbor_map,
                                                   neighbor_offset, transpose)
        else:
            # use the native pytorch XLA APIs for the TPU.
            cur_st = 0
            for kernel_idx in range(kernel.shape[0]):
                cur_ed = cur_st + neighbor_offset[kernel_idx]
                in_map = neighbor_map[cur_st:cur_ed, 0].long()
                out_map = neighbor_map[cur_st:cur_ed, 1].long()
                cur_st += neighbor_offset[kernel_idx]

                if transpose:
                    in_map, out_map = out_map, in_map

                out[out_map] += torch.mm(inputs[in_map], kernel[kernel_idx])

        ctx.for_backwards = (inputs, kernel, neighbor_map, neighbor_offset,
                             transpose)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        features, kernel, neighbor_map, neighbor_offset, transpose = ctx.for_backwards
        K, c_in, c_out = kernel.size()
        N_in = features.size(0)
        grad_features = torch.zeros(N_in, c_in, device=features.device)
        grad_kernel = torch.zeros(K, c_in, c_out, device=kernel.device)

        if 'cuda' in str(features.device):
            torchsparse_backend.sparseconv_backward(features, grad_features,
                                                    grad_out.contiguous(),
                                                    kernel, grad_kernel,
                                                    neighbor_map,
                                                    neighbor_offset, transpose)
        else:
            raise NotImplementedError
        return grad_features, grad_kernel, None, None, None, None


sparseconv_op = SpConvolution.apply


def conv3d(inputs: SparseTensor,
           weight: torch.Tensor,
           kernel_size: int,
           bias: Optional[torch.Tensor] = None,
           stride: int = 1,
           dilation: int = 1,
           transpose=False) -> SparseTensor:
    feats = inputs.feats
    coords = inputs.coords

    if kernel_size == 1:
        feats = feats.matmul(weight)
        if bias is not None:
            feats += bias
        outputs = SparseTensor(coords, feats, stride=inputs.stride)
        outputs.coord_maps = inputs.coord_maps
        outputs.kernel_maps = inputs.kernel_maps
        outputs.check()
    elif not transpose:
        kernel_map = inputs.kernel_maps.get(
            (kernel_size, inputs.stride, stride, dilation))

        if stride > 1:
            # do downsample
            kernel = KernelRegion(kernel_size, stride=inputs.stride)
            offsets = kernel.offsets.to(feats.device)
            new_coords = spdownsample(coords, stride * inputs.stride)
            hash_query = sphash(new_coords, offsets)
            hash_target = sphash(coords)
            idx_query = sphashquery(hash_query, hash_target)
            idx_query = list(convert_neighbor_map(idx_query))
            idx_query[1] = idx_query[1].to('cpu')
            sizes = (feats.shape[0], new_coords.shape[0])

            feats = sparseconv_op(feats, weight, idx_query[0], idx_query[1],
                                  sizes, transpose)
            if bias is not None:
                feats += bias
            outputs = SparseTensor(new_coords,
                                   feats,
                                   stride=inputs.stride * stride)
            outputs.coord_maps = copy.deepcopy(inputs.coord_maps)
            outputs.check()
            outputs.kernel_maps = copy.deepcopy(inputs.kernel_maps)
            outputs.kernel_maps[(kernel_size, inputs.stride, stride,
                                 dilation)] = idx_query + [sizes]

        else:
            # submanifold sparseconv
            if kernel_map is None:
                kernel = KernelRegion(kernel_size, stride=inputs.stride)
                offsets = kernel.offsets.to(feats.device)
                hash_query = sphash(coords, offsets)
                hash_target = sphash(coords)
                idx_query = sphashquery(hash_query, hash_target)
                idx_query = list(convert_neighbor_map(idx_query))
                idx_query[1] = idx_query[1].to('cpu')
                sizes = (feats.shape[0], feats.shape[0])

                feats = sparseconv_op(feats, weight, idx_query[0],
                                      idx_query[1], sizes, transpose)
                if bias is not None:
                    feats += bias
                outputs = SparseTensor(coords, feats, stride=inputs.stride)
                outputs.coord_maps = inputs.coord_maps
                outputs.check()
                outputs.kernel_maps = copy.deepcopy(inputs.kernel_maps)
                outputs.kernel_maps[(kernel_size, inputs.stride, stride,
                                     dilation)] = idx_query + [sizes]
            else:
                feats = sparseconv_op(feats, weight, kernel_map[0],
                                      kernel_map[1], kernel_map[2], transpose)
                if bias is not None:
                    feats += bias
                outputs = SparseTensor(coords, feats, stride=inputs.stride)
                outputs.coord_maps = inputs.coord_maps
                outputs.check()
                outputs.kernel_maps = inputs.kernel_maps
    else:
        # do upsample
        original_stride = int(inputs.stride / stride)
        kernel_map = inputs.kernel_maps.get(
            (kernel_size, original_stride, stride, dilation))
        feats = sparseconv_op(feats, weight, kernel_map[0], kernel_map[1],
                              kernel_map[2], transpose)
        if bias is not None:
            feats += bias
        outputs = SparseTensor(inputs.coord_maps[original_stride],
                               feats,
                               stride=original_stride)
        outputs.coord_maps = inputs.coord_maps
        outputs.kernel_maps = inputs.kernel_maps

    return outputs

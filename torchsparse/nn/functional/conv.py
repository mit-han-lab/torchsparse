import copy

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
                features,
                kernel,
                neighbor_map,
                neighbor_offset,
                sizes,
                transpose=False):
        features = features.contiguous()
        kernel = kernel.contiguous()
        if not transpose:
            out = torch.zeros(sizes[1],
                              kernel.size(-1),
                              device=features.device)
        else:
            # tbd: ensure the original, upsampled size to be the same.
            out = torch.zeros(sizes[0],
                              kernel.size(-1),
                              device=features.device)

        if 'cuda' in str(features.device):
            torchsparse_backend.sparseconv_forward(features, out, kernel,
                                                   neighbor_map,
                                                   neighbor_offset, transpose)
        #elif 'cpu' in str(features.device):
        #    torchsparse_backend.sparseconv_cpu_forward(features, out, kernel, neighbor_map, neighbor_offset.cpu(), transpose)
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
                # gather
                cur_feat = features[in_map]
                # gemm
                cur_feat = torch.mm(cur_feat, kernel[kernel_idx])
                # scatter
                out[out_map] += cur_feat

        ctx.for_backwards = (features, kernel, neighbor_map, neighbor_offset,
                             transpose)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        features, kernel, neighbor_map, neighbor_offset, transpose = ctx.for_backwards
        K, c_in, c_out = kernel.size()
        N_in = features.size(0)
        N_out = grad_out.size(0)
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
           weight,
           kernel_size,
           bias=None,
           stride: int = 1,
           dilation: int = 1,
           transpose=False) -> SparseTensor:
    feats = inputs.feats
    coords = inputs.coords

    if kernel_size == 1:
        feats = feats.matmul(weight)
        if bias is not None:
            feats += bias
        outputs = SparseTensor(coords=coords,
                               feats=feats,
                               stride=inputs.stride)
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
            idx_query = list(convert_neighbor_map_gpu(idx_query))
            idx_query[1] = idx_query[1].to('cpu')
            sizes = (feats.shape[0], new_coords.shape[0])

            feats = sparseconv_op(feats, weight, idx_query[0], idx_query[1],
                                  sizes, transpose)
            if bias is not None:
                feats += bias
            outputs = SparseTensor(coords=new_coords,
                                   feats=feats,
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
                idx_query = list(convert_neighbor_map_gpu(idx_query))
                idx_query[1] = idx_query[1].to('cpu')
                sizes = (feats.shape[0], feats.shape[0])

                feats = sparseconv_op(feats, weight, idx_query[0],
                                      idx_query[1], sizes, transpose)
                if bias is not None:
                    feats += bias
                outputs = SparseTensor(coords=coords,
                                       feats=feats,
                                       stride=inputs.stride)
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
                outputs = SparseTensor(coords=coords,
                                       feats=feats,
                                       stride=inputs.stride)
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
        outputs = SparseTensor(coords=inputs.coord_maps[original_stride],
                               feats=feats,
                               stride=original_stride)
        outputs.coord_maps = inputs.coord_maps
        outputs.kernel_maps = inputs.kernel_maps

    return outputs

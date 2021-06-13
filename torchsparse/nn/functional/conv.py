import copy

import torch
import torchsparse_backend
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd
from torchsparse import *
from torchsparse.nn.functional.squeeze_nmap import *
from torchsparse.nn.functional.downsample import *
from torchsparse.nn.functional.hash import *
from torchsparse.nn.functional.query import *
from torchsparse.utils.kernel_region import *

from typing import Union, List, Tuple

__all__ = ['conv3d']


class SpConvolution(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
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
                              dtype=features.dtype,
                              device=features.device)
        else:
            # tbd: ensure the original, upsampled size to be the same.
            out = torch.zeros(sizes[0],
                              kernel.size(-1),
                              dtype=features.dtype,
                              device=features.device)

        if 'cuda' in str(features.device):
            torchsparse_backend.sparseconv_forward(features, out, kernel,
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
    @custom_bwd
    def backward(ctx, grad_out):
        features, kernel, neighbor_map, neighbor_offset, transpose = ctx.for_backwards
        K, c_in, c_out = kernel.size()
        N_in = features.size(0)
        grad_features = torch.zeros(N_in,
                                    c_in,
                                    device=features.device,
                                    dtype=features.dtype)
        grad_kernel = torch.zeros(K,
                                  c_in,
                                  c_out,
                                  device=kernel.device,
                                  dtype=features.dtype)

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
           kernel: torch.Tensor,
           kernel_size: Union[int, List[int], Tuple[int, int, int]],
           bias: torch.Tensor = None,
           stride: Union[int, List[int], Tuple[int, int, int]] = 1,
           dilation: Union[int, List[int], Tuple[int, int, int]] = 1,
           transpose=False):
    features = inputs.F
    coords = inputs.C
    cur_stride = inputs.s
    
    # convert to hashable types
    if isinstance(kernel_size, list):
        kernel_size = tuple(kernel_size)
    elif isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(stride, list):
        stride = tuple(stride)
    elif isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(kernel_size, list):
        dilation = tuple(dilation)
    elif isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    if kernel_size == (1, 1, 1) and stride == (1, 1, 1) and dilation == (1, 1, 1):
        output_features = features.matmul(kernel)
        if bias is not None:
            output_features += bias
        output_tensor = SparseTensor(output_features, coords, cur_stride)
        output_tensor.coord_maps = inputs.coord_maps
        output_tensor.kernel_maps = inputs.kernel_maps
        output_tensor.check()
    elif not transpose:
        kernel_map_key = KernelMapConfig(kernel_size, cur_stride, stride, dilation)
        kernel_map = inputs.kernel_maps.get(kernel_map_key, None)

        do_downsample = False
        for i in range(3):
            if stride[i] > 1:
                do_downsample = True
                break

        if do_downsample:
            # do downsample
            kRegion = KernelRegion(kernel_size=kernel_size,
                                   tensor_stride=cur_stride)
            kOffset = kRegion.get_kernel_offset().to(features.device)
            new_coords = spdownsample(
                coords, 
                torch.IntTensor(stride).to(features.device).unsqueeze(0), 
                kernel_size, 
                torch.IntTensor(cur_stride).to(features.device).unsqueeze(0)
            )
            hash_query = sphash(new_coords, kOffset)
            hash_target = sphash(coords)
            idx_query = sphashquery(hash_query, hash_target)
            idx_query = list(squeeze_nmap(idx_query))
            idx_query[1] = idx_query[1].to('cpu')
            sizes = (features.shape[0], new_coords.shape[0])
            output_features = sparseconv_op(features, kernel, idx_query[0],
                                            idx_query[1], sizes, transpose)
            if bias is not None:
                output_features += bias
            output_tensor = SparseTensor(output_features, new_coords,
                                         [a * b for a, b in zip(cur_stride, stride)])
            output_tensor.coord_maps = copy.deepcopy(inputs.coord_maps)
            output_tensor.check()
            
            kernel_map_key = KernelMapConfig(kernel_size, cur_stride, stride,
                                       dilation)
            output_tensor.kernel_maps = copy.deepcopy(inputs.kernel_maps)
            output_tensor.kernel_maps[kernel_map_key] = idx_query + [sizes]

        else:
            if kernel_map is None:
                kRegion = KernelRegion(kernel_size=kernel_size,
                                       tensor_stride=cur_stride)
                try:
                    kOffset = kRegion.get_kernel_offset().to(features.device)
                except:
                    raise
                hash_query = sphash(coords, kOffset)
                hash_target = sphash(coords)
                idx_query = sphashquery(hash_query, hash_target)
                idx_query = list(squeeze_nmap(idx_query))
                idx_query[1] = idx_query[1].to('cpu')
                sizes = (features.shape[0], features.shape[0])
                output_features = sparseconv_op(features, kernel, idx_query[0],
                                                idx_query[1], sizes, transpose)
                if bias is not None:
                    output_features += bias
                output_tensor = SparseTensor(output_features, coords,
                                             cur_stride)
                output_tensor.coord_maps = inputs.coord_maps
                output_tensor.check()
                output_tensor.kernel_maps = copy.deepcopy(inputs.kernel_maps)
                kernel_map_key = KernelMapConfig(kernel_size, cur_stride, stride,
                                           dilation)
                output_tensor.kernel_maps[kernel_map_key] = idx_query + [sizes]
            else:
                output_features = sparseconv_op(features, kernel,
                                                kernel_map[0], kernel_map[1],
                                                kernel_map[2], transpose)
                if bias is not None:
                    output_features += bias
                output_tensor = SparseTensor(output_features, coords,
                                             cur_stride)
                output_tensor.coord_maps = inputs.coord_maps
                output_tensor.check()
                output_tensor.kernel_maps = inputs.kernel_maps

    else:
        # do upsample
                
        original_stride = tuple([int(a/b) for a, b in zip(cur_stride, stride)])
        
        kernel_map_key = KernelMapConfig(kernel_size, original_stride, stride, dilation)
        kernel_map = inputs.kernel_maps.get(kernel_map_key, None)
        assert kernel_map is not None, f'{kernel_map_key} does not exist.'
        output_features = sparseconv_op(features, kernel, kernel_map[0],
                                        kernel_map[1], kernel_map[2],
                                        transpose)
        if bias is not None:
            output_features += bias

        cur_coords = inputs.coord_maps.get(original_stride, None)
        assert cur_coords is not None, f'{original_stride} not in coord maps.'
        
        output_tensor = SparseTensor(output_features, cur_coords,
                                     original_stride)
        output_tensor.coord_maps = inputs.coord_maps
        output_tensor.check()
        output_tensor.kernel_maps = inputs.kernel_maps

    return output_tensor

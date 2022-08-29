from typing import List, Optional, Tuple, Union

import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

import torchsparse
import torchsparse.backend
from torchsparse import SparseTensor
from torchsparse.nn import functional as F
from torchsparse.utils import make_ntuple

buffer = torch.Tensor()

__all__ = ['conv3d']


class ConvolutionFunction(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        nbmaps: torch.Tensor,
        nbsizes: torch.Tensor,
        buffer: torch.Tensor,
        sizes: Tuple[int, int],
        input_mask: torch.Tensor,
        output_mask: torch.Tensor,
        epsilon: float,
        mm_thresh: int,
        conv_mode: int,
        transposed: bool = False,
    ) -> torch.Tensor:
        input = input.contiguous()
        weight = weight.contiguous()
        nbmaps = nbmaps.int().contiguous()
        nbsizes = nbsizes.int().contiguous()

        if not input.device.type == 'cuda':
            if not transposed:
                output = torch.zeros(sizes[1],
                                     weight.size(-1),
                                     dtype=input.dtype,
                                     device=input.device)
            else:
                # TODO(Haotian): ensure the original, upsampled size to be the same.
                output = torch.zeros(sizes[0],
                                     weight.size(-1),
                                     dtype=input.dtype,
                                     device=input.device)

        if input.device.type == 'cuda':
            output = torchsparse.backend.convolution_forward_cuda(
                input,
                weight,
                nbmaps,
                nbsizes.cpu(),
                input_mask,
                output_mask,
                sizes[1] if not transposed else sizes[0],
                epsilon,
                int(mm_thresh),
                conv_mode,
                transposed,
                buffer,
            )
        elif input.device.type == 'cpu':
            torchsparse.backend.convolution_forward_cpu(input, output, weight,
                                                        nbmaps, nbsizes.cpu(),
                                                        transposed)
        else:
            # use the native pytorch XLA APIs for the TPU.
            cur_st = 0
            for kernel_idx in range(weight.shape[0]):
                cur_ed = cur_st + nbsizes[kernel_idx]
                in_map = nbmaps[cur_st:cur_ed, 0].long()
                out_map = nbmaps[cur_st:cur_ed, 1].long()
                cur_st += nbsizes[kernel_idx]

                if transposed:
                    in_map, out_map = out_map, in_map

                cur_feat = input[in_map]
                cur_feat = torch.mm(cur_feat, weight[kernel_idx])
                output[out_map] += cur_feat
        ctx.for_backwards = (input, weight, nbmaps, nbsizes, transposed)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        input, weight, nbmaps, nbsizes, transposed = ctx.for_backwards

        grad_input = torch.zeros_like(input)
        grad_weight = torch.zeros_like(weight)

        if grad_output.device.type == 'cuda':
            torchsparse.backend.convolution_backward_cuda(
                input,
                grad_input,
                grad_output.contiguous(),
                weight,
                grad_weight,
                nbmaps,
                nbsizes.cpu(),
                transposed,
            )
        elif grad_output.device.type == 'cpu':
            torchsparse.backend.convolution_backward_cpu(
                input,
                grad_input,
                grad_output.contiguous(),
                weight,
                grad_weight,
                nbmaps,
                nbsizes.cpu(),
                transposed,
            )
        else:
            raise NotImplementedError
        return (
            grad_input,
            grad_weight,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def conv3d(
    input: SparseTensor,
    weight: torch.Tensor,
    kernel_size: Union[int, List[int], Tuple[int, ...]],
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, List[int], Tuple[int, ...]] = 1,
    dilation: Union[int, Tuple[int, ...]] = 1,
    transposed: bool = False,
    epsilon: float = 0.0,
    mm_thresh: int = 0,
    kmap_mode: str = 'hashmap',
) -> SparseTensor:
    feats, coords = input.feats, input.coords

    kernel_size = make_ntuple(kernel_size, ndim=3)
    stride = make_ntuple(stride, ndim=3)
    dilation = make_ntuple(dilation, ndim=3)

    conv_mode_num = 0
    global buffer
    if torchsparse.backends.benchmark:  # type: ignore
        conv_mode_num = 1 if (epsilon == 0.0 and mm_thresh == 0) else 2
        if buffer.shape[0] == 0 or buffer.dtype != input.F.dtype:
            buffer = torch.zeros(4000000 * 64,
                                 dtype=input.F.dtype,
                                 device=input.F.device,
                                 requires_grad=False)

    if kernel_size == (1, 1, 1) and stride == (1, 1, 1) and dilation == (1, 1,
                                                                         1):
        feats = feats.matmul(weight)
        if bias is not None:
            feats += bias
        output = SparseTensor(coords=coords, feats=feats, stride=input.stride)
    elif not transposed:
        kmap = input.kmaps.get((input.stride, kernel_size, stride, dilation))
        if kmap is None:
            if any(s > 1 for s in stride):
                kmap_out = F.build_kernel_map(coords, kernel_size, stride,
                                              input.stride, kmap_mode)
                if len(kmap_out) == 3:
                    nbmaps, nbsizes, coords = kmap_out
                    input_mask, output_mask = None, None
                elif len(kmap_out) == 5:
                    nbmaps, nbsizes, coords, input_mask, output_mask = kmap_out
                else:
                    raise NotImplementedError
            else:

                kmap_out = F.build_kernel_map(coords, kernel_size, stride,
                                              input.stride, kmap_mode)
                if len(kmap_out) == 2:
                    nbmaps, nbsizes = kmap_out
                    input_mask, output_mask = None, None
                elif len(kmap_out) == 4:
                    nbmaps, nbsizes, input_mask, output_mask = kmap_out
                else:
                    raise NotImplementedError

            nbsizes = nbsizes.cpu()

            kmap = [
                nbmaps,
                nbsizes,
                (feats.shape[0], coords.shape[0]),
                input_mask,
                output_mask,
            ]
            input.kmaps[(input.stride, kernel_size, stride, dilation)] = kmap

        feats = ConvolutionFunction.apply(
            feats,
            weight,
            kmap[0],
            kmap[1],
            buffer,
            kmap[2],
            kmap[3],
            kmap[4],
            epsilon,
            mm_thresh,
            conv_mode_num,
            transposed,
        )
        if bias is not None:
            feats += bias
        output = SparseTensor(
            coords=coords,
            feats=feats,
            stride=tuple(input.stride[k] * stride[k] for k in range(3)),
        )
    else:
        tensor_stride = tuple(input.stride[k] // stride[k] for k in range(3))
        kmap = input.kmaps[(tensor_stride, kernel_size, stride, dilation)]
        feats = ConvolutionFunction.apply(
            feats,
            weight,
            kmap[0],
            kmap[1],
            buffer,
            kmap[2],
            kmap[3],
            kmap[4],
            epsilon,
            mm_thresh,
            conv_mode_num,
            transposed,
        )
        if bias is not None:
            feats += bias
        output = SparseTensor(
            coords=input.cmaps[tensor_stride],
            feats=feats,
            stride=tensor_stride,
        )

    output.cmaps = input.cmaps
    output.cmaps.setdefault(output.stride, output.coords)
    output.kmaps = input.kmaps
    return output

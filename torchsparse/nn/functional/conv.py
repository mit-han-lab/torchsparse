from typing import Optional, Tuple, Union

import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

import torchsparse.backend
from torchsparse import SparseTensor
from torchsparse.nn import functional as F
from torchsparse.nn.utils import get_kernel_offsets
from torchsparse.utils import make_ntuple

__all__ = ['conv3d']


class ConvolutionFunction(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(ctx,
                input: torch.Tensor,
                weight: torch.Tensor,
                nbmaps: torch.Tensor,
                nbsizes: torch.Tensor,
                sizes: Tuple[int, int],
                transposed: bool = False) -> torch.Tensor:
        input = input.contiguous()
        weight = weight.contiguous()
        nbmaps = nbmaps.int().contiguous()
        nbsizes = nbsizes.int().contiguous()

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
            torchsparse.backend.convolution_forward_cuda(
                input, output, weight, nbmaps, nbsizes.cpu(), transposed)
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
                input, grad_input, grad_output.contiguous(), weight,
                grad_weight, nbmaps, nbsizes.cpu(), transposed)
        else:
            raise NotImplementedError
        return grad_input, grad_weight, None, None, None, None


def conv3d(input: SparseTensor,
           weight: torch.Tensor,
           kernel_size: Union[int, Tuple[int, ...]],
           bias: Optional[torch.Tensor] = None,
           stride: Union[int, Tuple[int, ...]] = 1,
           dilation: Union[int, Tuple[int, ...]] = 1,
           transposed: bool = False) -> SparseTensor:
    feats, coords = input.feats, input.coords

    kernel_size = make_ntuple(kernel_size, ndim=3)
    stride = make_ntuple(stride, ndim=3)
    dilation = make_ntuple(dilation, ndim=3)

    if (kernel_size == (1, 1, 1) and stride == (1, 1, 1)
            and dilation == (1, 1, 1)):
        feats = feats.matmul(weight)
        if bias is not None:
            feats += bias
        output = SparseTensor(coords=coords, feats=feats, stride=input.stride)
    elif not transposed:
        kmap = input.kmaps.get((input.stride, kernel_size, stride, dilation))
        if kmap is None:
            offsets = get_kernel_offsets(kernel_size,
                                         stride=input.stride,
                                         device=feats.device)

            references = F.sphash(coords)
            if any(s > 1 for s in stride):
                coords = F.spdownsample(coords, stride, kernel_size,
                                        input.stride)
            queries = F.sphash(coords, offsets)
            results = F.sphashquery(queries, references)

            nbsizes = torch.sum(results != -1, dim=1)
            nbmaps = torch.nonzero(results != -1)
            nbmaps[:, 0] = results.view(-1)[nbmaps[:, 0] * results.size(1)
                                            + nbmaps[:, 1]]

            kmap = [nbmaps, nbsizes, (feats.shape[0], coords.shape[0])]
            input.kmaps[(input.stride, kernel_size, stride, dilation)] = kmap

        feats = ConvolutionFunction.apply(feats, weight, kmap[0], kmap[1],
                                          kmap[2], transposed)
        if bias is not None:
            feats += bias
        output = SparseTensor(
            coords=coords,
            feats=feats,
            stride=tuple(input.stride[k] * stride[k] for k in range(3)))
    else:
        tensor_stride = tuple(input.stride[k] // stride[k] for k in range(3))
        kmap = input.kmaps[(tensor_stride, kernel_size, stride, dilation)]

        feats = ConvolutionFunction.apply(feats, weight, kmap[0], kmap[1],
                                          kmap[2], transposed)
        if bias is not None:
            feats += bias
        output = SparseTensor(coords=input.cmaps[tensor_stride],
                              feats=feats,
                              stride=tensor_stride)

    output.cmaps = input.cmaps
    output.cmaps.setdefault(output.stride, output.coords)
    output.kmaps = input.kmaps
    return output

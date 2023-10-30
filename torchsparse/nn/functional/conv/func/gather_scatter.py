from typing import Dict

import torch
from torch.autograd import Function

# from torch.cuda.amp import custom_bwd, custom_fwd

import torchsparse
import torchsparse.backend

buffer = torch.Tensor()

__all__ = ["GatherScatterConvolutionFuntion"]


class GatherScatterConvolutionFuntion(Function):  # TorchSparse_v2
    @staticmethod
    # @custom_fwd(cast_inputs=torch.half)
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        kmap: Dict,
        config: Dict,
        transposed: bool = False,
    ) -> torch.Tensor:
        nbmaps = kmap["nbmaps"]
        nbsizes = kmap["nbsizes"].cpu()
        sizes = kmap["sizes"]
        input_mask = kmap["input_mask"]
        output_mask = kmap["output_mask"]
        epsilon = config["epsilon"]
        mm_thresh = config["mm_thresh"]

        conv_mode = 0
        global buffer
        if torchsparse.backends.benchmark:  # type: ignore
            conv_mode = 1 if (epsilon == 0.0 and mm_thresh == 0) else 2
            if buffer.shape[0] == 0 or buffer.dtype != input.dtype:
                buffer = torch.zeros(
                    4000000 * 64,
                    dtype=input.dtype,
                    device=input.device,
                    requires_grad=False,
                )

        input = input.contiguous()
        weight = weight.contiguous()
        nbmaps = nbmaps.int().contiguous()
        nbsizes = nbsizes.int().contiguous()

        if not input.device.type == "cuda":
            if not transposed:
                output = torch.zeros(
                    sizes[1], weight.size(-1), dtype=input.dtype, device=input.device
                )
            else:
                # TODO(Haotian): ensure the original, upsampled size to be the same.
                output = torch.zeros(
                    sizes[0], weight.size(-1), dtype=input.dtype, device=input.device
                )

        if input.device.type == "cuda":
            if torch.float16 in [input.dtype, weight.dtype]:
                input = input.to(torch.float16)
                weight = weight.to(torch.float16)

            output = torchsparse.backend.conv_forward_gather_scatter_cuda(
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
        elif input.device.type == "cpu":
            torchsparse.backend.conv_forward_gather_scatter_cpu(
                input, output, weight, nbmaps, nbsizes.cpu(), transposed
            )
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
        return output.to(weight.dtype)

    @staticmethod
    # @custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        input, weight, nbmaps, nbsizes, transposed = ctx.for_backwards

        if grad_output.dtype != weight.dtype:
            grad_output = grad_output.to(weight.dtype)

        grad_input = torch.zeros_like(input)
        grad_weight = torch.zeros_like(weight)

        if grad_output.device.type == "cuda":
            torchsparse.backend.conv_backward_gather_scatter_cuda(
                input,
                grad_input,
                grad_output.contiguous(),
                weight,
                grad_weight,
                nbmaps,
                nbsizes.cpu(),
                transposed,
            )
        elif grad_output.device.type == "cpu":
            torchsparse.backend.conv_backward_gather_scatter_cpu(
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
        )

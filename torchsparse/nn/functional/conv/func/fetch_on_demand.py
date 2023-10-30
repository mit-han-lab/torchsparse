from typing import Dict

import torch
from torch.autograd import Function

# from torch.cuda.amp import custom_bwd, custom_fwd

import torchsparse
import torchsparse.backend

# TODO: Fetch_on_demand do not have backward kernels now.
#       Using Gather_Scatter for backward propogation.

__all__ = ["FetchOnDemandConvolutionFuntion"]


class FetchOnDemandConvolutionFuntion(Function):
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

        """if transposed:
            input_nbmaps = kmap["nbmaps"][1, :]
            output_nbmaps = kmap["nbmaps"][0, :]
        else:
            input_nbmaps = kmap["nbmaps"][0, :]
            output_nbmaps = kmap["nbmaps"][1, :]

        M = nbmaps.size(0)
        nbmaps_t = torch.zeros((2, M),
            dtype=torch.int, device=input.device, requires_grad=False)
        for l in range(M):
            nbmaps_t[0, l] = nbmaps[l, 0]
            nbmaps_t[1, l] = nbmaps[l, 1]"""

        nbmaps = kmap["nbmaps"]
        nbsizes = kmap["nbsizes"]
        nbaddrs = kmap["nbaddrs"]
        qnbaddrs = kmap["qnbaddrs"]
        sizes = kmap["sizes"]
        qmapsize = kmap["qmapsize"]

        mapsize = nbmaps.size(1)

        input = input.contiguous()
        weight = weight.contiguous()

        # nbmaps = nbmaps.int().contiguous()
        # input_nbmaps = input_nbmaps.int().contiguous()
        # output_nbmaps = output_nbmaps.int().contiguous()
        # nbaddrs = nbaddrs.int().contiguous()
        # qnbaddrs = qnbaddrs.int().contiguous()
        # nbsizes = nbsizes.int().contiguous()

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

            if config["FOD_fusion"] == True:
                output = torchsparse.backend.conv_forward_fetch_on_demand_cuda(
                    input,
                    weight,
                    nbmaps,
                    mapsize,
                    nbaddrs,
                    qnbaddrs,
                    sizes[1] if not transposed else sizes[0],
                    qmapsize,
                    transposed,
                    torchsparse.backends.allow_tf32,
                    torchsparse.backends.allow_fp16,
                )
            else:
                output = (
                    torchsparse.backend.conv_forward_fetch_on_demand_no_fusion_cuda(
                        input,
                        weight,
                        nbmaps,
                        nbsizes.cpu(),
                        mapsize,
                        sizes[1] if not transposed else sizes[0],
                        transposed,
                        torchsparse.backends.allow_tf32,
                        torchsparse.backends.allow_fp16,
                    )
                )

        else:
            raise NotImplementedError

        ctx.for_backwards = (input, weight, nbmaps, nbsizes, transposed)
        return output.to(weight.dtype)

    @staticmethod
    # @custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        input, weight, nbmaps, nbsizes, transposed = ctx.for_backwards

        if grad_output.dtype != weight.dtype:
            grad_output = grad_output.to(weight.dtype)

        print(
            "[Warning] Fetch_On_Demand does not have backward kernels now. Use Gather-Scatter for backward."
        )
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
        return (grad_input, grad_weight, None, None, None, None)

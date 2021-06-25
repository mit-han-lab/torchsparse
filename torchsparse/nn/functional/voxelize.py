import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

import torchsparse.backend

__all__ = ['spvoxelize']


class VoxelizeFunction(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(ctx, feats: torch.Tensor, coords: torch.Tensor,
                counts: torch.Tensor) -> torch.Tensor:
        feats = feats.contiguous()
        coords = coords.contiguous().int()

        if feats.device.type == 'cuda':
            output = torchsparse.backend.voxelize_forward_cuda(
                feats, coords, counts)
        elif feats.device.type == 'cpu':
            output = torchsparse.backend.voxelize_forward_cpu(
                feats, coords, counts)
        else:
            device = feats.device
            output = torchsparse.backend.voxelize_forward_cpu(
                feats.cpu(), coords.cpu(), counts.cpu()).to(device)

        ctx.for_backwards = (coords, counts, feats.shape[0])
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        coords, counts, input_size = ctx.for_backwards
        grad_output = grad_output.contiguous()

        if grad_output.device.type == 'cuda':
            grad_feats = torchsparse.backend.voxelize_backward_cuda(
                grad_output, coords, counts, input_size)
        elif grad_output.device.type == 'cpu':
            grad_feats = torchsparse.backend.voxelize_backward_cpu(
                grad_output, coords, counts, input_size)
        else:
            device = grad_output.device
            grad_feats = torchsparse.backend.voxelize_backward_cpu(
                grad_output.cpu(), coords.cpu(), counts.cpu(),
                input_size).to(device)

        return grad_feats, None, None


def spvoxelize(feats: torch.Tensor, coords: torch.Tensor,
               counts: torch.Tensor) -> torch.Tensor:
    return VoxelizeFunction.apply(feats, coords, counts)

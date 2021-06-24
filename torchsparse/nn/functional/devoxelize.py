import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

import torchsparse.backend

__all__ = ['spdevoxelize', 'calc_ti_weights']


def calc_ti_weights(coords: torch.Tensor,
                    idx_query: torch.Tensor,
                    scale: float = 1) -> torch.Tensor:
    with torch.no_grad():
        p = coords
        if scale != 1:
            pf = torch.floor(coords / scale) * scale
        else:
            pf = torch.floor(coords)
        pc = pf + scale

        x = p[:, 0].view(-1, 1)
        y = p[:, 1].view(-1, 1)
        z = p[:, 2].view(-1, 1)

        xf = pf[:, 0].view(-1, 1).float()
        yf = pf[:, 1].view(-1, 1).float()
        zf = pf[:, 2].view(-1, 1).float()

        xc = pc[:, 0].view(-1, 1).float()
        yc = pc[:, 1].view(-1, 1).float()
        zc = pc[:, 2].view(-1, 1).float()

        w0 = (xc - x) * (yc - y) * (zc - z)
        w1 = (xc - x) * (yc - y) * (z - zf)
        w2 = (xc - x) * (y - yf) * (zc - z)
        w3 = (xc - x) * (y - yf) * (z - zf)
        w4 = (x - xf) * (yc - y) * (zc - z)
        w5 = (x - xf) * (yc - y) * (z - zf)
        w6 = (x - xf) * (y - yf) * (zc - z)
        w7 = (x - xf) * (y - yf) * (z - zf)

        w = torch.cat([w0, w1, w2, w3, w4, w5, w6, w7], dim=1)
        w = w.transpose(1, 0).contiguous()
        if scale != 1:
            w /= scale ** 3
        w[idx_query == -1] = 0
        w /= torch.sum(w, dim=0) + 1e-8
    return w


class DevoxelizeFunction(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(ctx, feats: torch.Tensor, coords: torch.Tensor,
                weights: torch.Tensor) -> torch.Tensor:
        feats = feats.contiguous()
        coords = coords.contiguous().int()
        weights = weights.contiguous()

        if feats.device.type == 'cuda':
            output = torchsparse.backend.devoxelize_forward_cuda(
                feats, coords, weights)
        elif feats.device.type == 'cpu':
            output = torchsparse.backend.devoxelize_forward_cpu(
                feats, coords, weights)
        else:
            device = feats.device
            output = torchsparse.backend.devoxelize_forward_cpu(
                feats.cpu(), coords.cpu(), weights.cpu()).to(device)

        ctx.for_backwards = (coords, weights, feats.shape[0])
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        coords, weights, input_size = ctx.for_backwards
        grad_output = grad_output.contiguous()

        if grad_output.device.type == 'cuda':
            grad_feats = torchsparse.backend.devoxelize_backward_cuda(
                grad_output, coords, weights, input_size)
        elif grad_output.device.type == 'cpu':
            grad_feats = torchsparse.backend.devoxelize_backward_cpu(
                grad_output, coords, weights, input_size)
        else:
            device = grad_output.device
            grad_feats = torchsparse.backend.devoxelize_backward_cpu(
                grad_output.cpu(), coords.cpu(), weights.cpu(),
                input_size).to(device)

        return grad_feats, None, None


def spdevoxelize(feats: torch.Tensor, coords: torch.Tensor,
                 weights: torch.Tensor) -> torch.Tensor:
    return DevoxelizeFunction.apply(feats, coords, weights)

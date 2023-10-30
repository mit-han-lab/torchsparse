import torch

from torchsparse import SparseTensor

__all__ = ["global_avg_pool", "global_max_pool"]


def global_avg_pool(inputs: SparseTensor) -> torch.Tensor:
    batch_size = torch.max(inputs.coords[:, 0]).item() + 1
    outputs = []
    for k in range(batch_size):
        input = inputs.feats[inputs.coords[:, 0] == k]
        output = torch.mean(input, dim=0)
        outputs.append(output)
    outputs = torch.stack(outputs, dim=0)
    return outputs


def global_max_pool(inputs: SparseTensor) -> torch.Tensor:
    batch_size = torch.max(inputs.coords[:, 0]).item() + 1
    outputs = []
    for k in range(batch_size):
        input = inputs.feats[inputs.coords[:, 0] == k]
        output = torch.max(input, dim=0)[0]
        outputs.append(output)
    outputs = torch.stack(outputs, dim=0)
    return outputs

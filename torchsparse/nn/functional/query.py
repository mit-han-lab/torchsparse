import torch

import torchsparse.backend

__all__ = ['sphashquery']


def sphashquery(queries: torch.Tensor,
                references: torch.Tensor) -> torch.Tensor:
    queries = queries.contiguous()
    references = references.contiguous()

    sizes = queries.size()
    queries = queries.view(-1)

    indices = torch.arange(len(references),
                           device=queries.device,
                           dtype=torch.long)

    if queries.device.type == 'cuda':
        output = torchsparse.backend.hash_query_cuda(queries, references,
                                                     indices)
    elif queries.device.type == 'cpu':
        output = torchsparse.backend.hash_query_cpu(queries, references,
                                                    indices)
    else:
        device = queries.device
        output = torchsparse.backend.hash_query_cpu(queries.cpu(),
                                                    references.cpu(),
                                                    indices.cpu()).to(device)

    output = (output - 1).view(*sizes)
    return output

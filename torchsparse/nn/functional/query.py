import torch

import torchsparse.backend

__all__ = ["sphashquery"]


def sphashquery(queries: torch.Tensor, references: torch.Tensor) -> torch.Tensor:
    queries = queries.contiguous()
    references = references.contiguous()

    sizes = queries.size()
    queries = queries.view(-1)

    hashmap_keys = torch.zeros(
        2 * references.shape[0], dtype=torch.int64, device=references.device
    )
    hashmap_vals = torch.zeros(
        2 * references.shape[0], dtype=torch.int32, device=references.device
    )
    hashmap = torchsparse.backend.GPUHashTable(hashmap_keys, hashmap_vals)
    hashmap.insert_vals(references)

    if queries.device.type == "cuda":
        output = hashmap.lookup_vals(queries)[: queries.shape[0]]
    elif queries.device.type == "cpu":
        indices = torch.arange(len(references), device=queries.device, dtype=torch.long)
        output = torchsparse.backend.hash_query_cpu(queries, references, indices)
    else:
        device = queries.device
        indices = torch.arange(len(references), device=queries.device, dtype=torch.long)
        output = torchsparse.backend.hash_query_cpu(
            queries.cpu(), references.cpu(), indices.cpu()
        ).to(device)

    output = (output - 1).view(*sizes)
    return output

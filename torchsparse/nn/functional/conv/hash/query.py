import torch

import torchsparse.backend

__all__ = ["sphashquery", "convert_transposed_out_in_map"]


def sphashquery(queries: torch.Tensor, references: torch.Tensor) -> torch.Tensor:
    queries = queries.contiguous()
    references = references.contiguous()

    sizes = queries.size()
    queries = queries.view(-1)

    indices = torch.arange(len(references), device=queries.device, dtype=torch.long)

    if queries.device.type == "cuda":
        hashtable = torchsparse.backend.GPUHashTable(references.shape[0] * 2)
        hashtable.insert_vals(references)
        output = hashtable.lookup_vals(queries)
    elif queries.device.type == "cpu":
        output = torchsparse.backend.hash_query_cpu(queries, references, indices)
    else:
        device = queries.device
        output = torchsparse.backend.hash_query_cpu(
            queries.cpu(), references.cpu(), indices.cpu()
        ).to(device)

    output = (output - 1).view(*sizes)
    if output.shape[0] % 128 != 0:
        output = torch.cat(
            [
                output,
                torch.zeros(
                    128 - output.shape[0] % 128,
                    output.shape[1],
                    device=output.device,
                    dtype=output.dtype,
                )
                - 1,
            ],
            dim=0,
        )
    return output


def convert_transposed_out_in_map(out_in_map, size):
    out_in_map_t = torch.full(
        (size, out_in_map.shape[1]),
        fill_value=-1,
        device=out_in_map.device,
        dtype=torch.int32,
    )
    torchsparse.backend.convert_transposed_out_in_map(out_in_map, out_in_map_t)
    return out_in_map_t

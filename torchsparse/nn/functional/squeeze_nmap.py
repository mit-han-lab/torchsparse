import torch

__all__ = ['squeeze_nmap']


def squeeze_nmap(neighbor_map):
    idx_batch, idx_point = torch.where(neighbor_map != -1)
    map_converted = neighbor_map.view(-1)[idx_batch * neighbor_map.size(1) +
                                          idx_point]
    map_converted = torch.stack([map_converted, idx_point], dim=1)
    nmap_offset = torch.sum(neighbor_map != -1, 1)
    return map_converted.int().contiguous(), nmap_offset.int().contiguous()

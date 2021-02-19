import torch
import torchsparse_backend


def convert_neighbor_map(neighbor_map):
    idx_batch, idx_point = torch.where(neighbor_map != -1)
    if neighbor_map.device.type == 'cuda':
        map_converted = torchsparse_backend.convert_map_forward(
            neighbor_map.int(), idx_batch.int(), idx_point.int())
    elif neighbor_map.device.type == 'cpu':
        map_converted = torchsparse_backend.cpu_convert_map_forward(
            neighbor_map.int(), idx_batch.int(), idx_point.int())
    else:
        device = neighbor_map.device
        map_converted = torchsparse_backend.cpu_convert_map_forward(
            neighbor_map.int().cpu(),
            idx_batch.int().cpu(),
            idx_point.int().cpu())
        map_converted = map_converted.to(device)
    nmap_offset = torch.sum(neighbor_map != -1, 1)
    return map_converted.int().contiguous(), nmap_offset.int().contiguous()

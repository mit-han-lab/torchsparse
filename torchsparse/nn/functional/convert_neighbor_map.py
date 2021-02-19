import torch
import torchsparse_cuda
from torch.autograd import Function


class ConvertNeighborMap(Function):
    @staticmethod
    def forward(ctx, neighbor_map):
        idx_batch, idx_point = torch.where(neighbor_map != -1)
        if 'cuda' in str(neighbor_map.device):
            map_converted = torchsparse_cuda.convert_map_forward(
                neighbor_map.int(), idx_batch.int(), idx_point.int())
        elif 'cpu' in str(neighbor_map.device):
            map_converted = torchsparse_cuda.cpu_convert_map_forward(
                neighbor_map.int(), idx_batch.int(), idx_point.int())
        else:
            device = neighbor_map.device
            map_converted = torchsparse_cuda.cpu_convert_map_forward(
                neighbor_map.int().cpu(),
                idx_batch.int().cpu(),
                idx_point.int().cpu())
            map_converted = map_converted.to(device)
        nmap_offset = torch.sum(neighbor_map != -1, 1)
        return map_converted.int().contiguous(), nmap_offset.int().contiguous()


convert_neighbor_map_gpu = ConvertNeighborMap.apply

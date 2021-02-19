#include <torch/torch.h>
#include "convert_neighbor_map_gpu.h"
#include <vector>

at::Tensor convert_map_forward(
    const at::Tensor nmap,
    const at::Tensor idx_batch,
    const at::Tensor idx_point)
{
  //return group_point_forward_gpu(points, indices);

  int N = nmap.size(1);
  int k = nmap.size(0);
  int N_nonzero = idx_point.size(0);
  at::Tensor out = torch::zeros({N_nonzero, 2}, at::device(nmap.device()).dtype(at::ScalarType::Int));
  convert_map_wrapper(k, N, N_nonzero, nmap.data_ptr<int>(), idx_batch.data_ptr<int>(), idx_point.data_ptr<int>(), out.data_ptr<int>());
  return out;
}

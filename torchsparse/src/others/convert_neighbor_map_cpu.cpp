#include <torch/torch.h>
#include "convert_neighbor_map_cpu_header.h"
#include <vector>

void cpu_convert_map_wrapper(int k, int N, int N_nonzero, const int *nmap, const int *idx_batch, const int *idx_point, int *out)
{
#pragma omp parallel for
  for (int index = 0; index < N_nonzero; index++)
  {
    int i = idx_batch[index];
    int j = idx_point[index];
    out[index << 1] = nmap[i * N + j];
    out[(index << 1) + 1] = j;
  }
}

at::Tensor cpu_convert_map_forward(
    const at::Tensor nmap,
    const at::Tensor idx_batch,
    const at::Tensor idx_point)
{
  //return group_point_forward_gpu(points, indices);

  int N = nmap.size(1);
  int k = nmap.size(0);
  int N_nonzero = idx_point.size(0);
  at::Tensor out = torch::zeros({N_nonzero, 2}, at::device(nmap.device()).dtype(at::ScalarType::Int));
  cpu_convert_map_wrapper(k, N, N_nonzero, nmap.data_ptr<int>(), idx_batch.data_ptr<int>(), idx_point.data_ptr<int>(), out.data_ptr<int>());
  return out;
}

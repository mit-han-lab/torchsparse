#include <torch/torch.h>
#include <vector>
#include "count_gpu.h"

at::Tensor count_forward_cuda(
    const at::Tensor idx,
    const int s)
{
  int N = idx.size(0);
  at::Tensor out = torch::zeros({s}, at::device(idx.device()).dtype(at::ScalarType::Int));
  count_wrapper(N, idx.data_ptr<int>(), out.data_ptr<int>());
  return out;
}

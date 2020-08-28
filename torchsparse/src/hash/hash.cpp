#include <torch/torch.h>
#include <vector>
#include "hash_gpu.h"

at::Tensor hash_forward(
    const at::Tensor idx
)
{  
  int N = idx.size(0);
  at::Tensor out = torch::zeros({N}, at::device(idx.device()).dtype(at::ScalarType::Long));
  hash_wrapper(N, idx.data_ptr<int>(), out.data_ptr<long>());
  return out;
}


at::Tensor kernel_hash_forward(
    const at::Tensor idx,
    const at::Tensor kernel_offset
)
{  
  int N = idx.size(0);
  int K = kernel_offset.size(0);
  at::Tensor out = torch::zeros({K, N}, at::device(idx.device()).dtype(at::ScalarType::Long));
  kernel_hash_wrapper(N, K, idx.data_ptr<int>(), kernel_offset.data_ptr<int>(), out.data_ptr<long>());
  return out;
}



/*
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hash_forward", &hash_forward, "Hashing forward (CUDA)");
  m.def("kernel_hash_forward", &kernel_hash_forward, "Kernel Hashing forward (CUDA)");
}
*/


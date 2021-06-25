#include <stdio.h>
#include <stdlib.h>
#include <torch/torch.h>

#include <cmath>
#include <vector>

// counting
// input N*3 int32 tensor output N*1 int64 tensor
__global__ void count_kernel(int N, const int *__restrict__ data,
                             int *__restrict__ out) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N && data[i] >= 0) {
    atomicAdd(&out[data[i]], 1);
  }
}

void count_wrapper(int N, const int *data, int *out) {
  count_kernel<<<ceil((double)N / 512), 512>>>(N, data, out);
}

// make sure indices is int type
// feat: (b,c,n) indices: (b,n) -> out: (b,c,s), out_indices: (b,n)
// (preprocessed indices)
at::Tensor count_cuda(const at::Tensor idx, const int s) {
  int N = idx.size(0);
  at::Tensor out =
      torch::zeros({s}, at::device(idx.device()).dtype(at::ScalarType::Int));
  count_wrapper(N, idx.data_ptr<int>(), out.data_ptr<int>());
  return out;
}

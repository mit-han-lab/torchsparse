#include "voxelize_cpu.h"

#include <torch/torch.h>

#include <vector>

at::Tensor voxelize_forward_cpu(const at::Tensor inputs, const at::Tensor idx,
                                const at::Tensor counts) {
  int N = inputs.size(0);
  int c = inputs.size(1);
  int N1 = counts.size(0);
  at::Tensor out = torch::zeros(
      {N1, c}, at::device(idx.device()).dtype(at::ScalarType::Float));
  for (int i = 0; i < N; i++) {
    int pos = *(idx.data_ptr<int>() + i);
    if (*(counts.data_ptr<int>() + pos) == 0) continue;
#pragma omp parallel for
    for (int j = 0; j < c; j++) {
      *(out.data_ptr<float>() + pos * c + j) +=
          *(inputs.data_ptr<float>() + i * c + j) /
          (float)(*(counts.data_ptr<int>() + pos));
    }
  }
  return out;
}

at::Tensor voxelize_backward_cpu(const at::Tensor top_grad,
                                 const at::Tensor idx, const at::Tensor counts,
                                 const int N) {
  int c = top_grad.size(1);
  at::Tensor bottom_grad = torch::zeros(
      {N, c}, at::device(idx.device()).dtype(at::ScalarType::Float));
  for (int i = 0; i < N; i++) {
    if (*(counts.data_ptr<int>() + *(idx.data_ptr<int>() + i)) == 0) continue;
#pragma omp parallel for
    for (int j = 0; j < c; j++) {
      *(bottom_grad.data_ptr<float>() + i * c + j) =
          *(top_grad.data_ptr<float>() + *(idx.data_ptr<int>() + i) * c + j) /
          (float)(*(counts.data_ptr<int>() + *(idx.data_ptr<int>() + i)));
    }
  }
  return bottom_grad;
}

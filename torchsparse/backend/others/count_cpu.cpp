#include "count_cpu.h"

#include <torch/torch.h>

#include <vector>

at::Tensor count_cpu(const at::Tensor idx, const int s) {
  int N = idx.size(0);
  at::Tensor out =
      torch::zeros({s}, at::device(idx.device()).dtype(at::ScalarType::Int));
  int *idx_ = idx.data_ptr<int>();
  int *out_ = out.data_ptr<int>();
#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    int cur_idx = idx_[i];
    if (cur_idx < 0) {
      continue;
    }
#pragma omp atomic
    out_[cur_idx]++;
  }
  return out;
}

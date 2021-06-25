#include "devoxelize_cpu.h"

#include <torch/torch.h>

#include <vector>

// make sure indices is int type
// feat: (b,c,s) indices: (N, 3) batch_index: (N, ) -> out: (N, c)
at::Tensor devoxelize_forward_cpu(const at::Tensor feat,
                                  const at::Tensor indices,
                                  const at::Tensor weight) {
  int c = feat.size(1);
  int N = indices.size(0);

  at::Tensor out = torch::zeros(
      {N, c}, at::device(feat.device()).dtype(at::ScalarType::Float));
#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    int *indices_ = indices.data_ptr<int>() + i * 8;
    float *weight_ = weight.data_ptr<float>() + i * 8;
    for (int j = 0; j < c; j++) {
      float *feat_ = feat.data_ptr<float>() + j;
      float cur_feat;
      for (int k = 0; k < 8; k++) {
        cur_feat = (indices_[k] >= 0) ? feat_[indices_[k] * c] : 0;
        *(out.data_ptr<float>() + i * c + j) += weight_[k] * cur_feat;
      }
    }
  }
  return out;
}

// top_grad: (N, c), indices: (N, 3), batch_index: (N, ) -> bottom_grad:
// (b,c,s), s=r^3
at::Tensor devoxelize_backward_cpu(const at::Tensor top_grad,
                                   const at::Tensor indices,
                                   const at::Tensor weight, int n) {
  int c = top_grad.size(1);
  int N = top_grad.size(0);
  at::Tensor bottom_grad = torch::zeros(
      {n, c}, at::device(top_grad.device()).dtype(at::ScalarType::Float));

  for (int i = 0; i < N; i++) {
    int *indices_ = indices.data_ptr<int>() + i * 8;
    float *weight_ = weight.data_ptr<float>() + i * 8;
#pragma omp parallel for
    for (int j = 0; j < c; j++) {
      float *top_grad_ = top_grad.data_ptr<float>() + j;
      float cur_top_grad;
      for (int k = 0; k < 8; k++) {
        cur_top_grad = (indices_[k] >= 0) ? top_grad_[indices_[k] * c] : 0;
        *(bottom_grad.data_ptr<float>() + indices_[k] * c + j) +=
            weight_[k] * cur_top_grad;
      }
    }
  }

  return bottom_grad;
}

#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_vector.h>
#include <torch/extension.h>

#include <THC/THCAtomics.cuh>

// input features (n, c), indices (N, 8), weight (N, 8) -> output features (N,
// c)
template <typename scalar_t>
__global__ void devoxelize_forward_kernel(int N, int c,
                                          const int *__restrict__ indices,
                                          const scalar_t *__restrict__ weight,
                                          const scalar_t *__restrict__ feat,
                                          scalar_t *__restrict__ out) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int i = index / c;
  int j = index % c;

  if (i < N) {
    const int *indices_ = indices + 8 * i;
    const scalar_t *weight_ = weight + 8 * i;
    const scalar_t *feat_ = feat + j;

    scalar_t cur_feat;
    for (int k = 0; k < 8; k++) {
      cur_feat = 0;
      if (indices_[k] >= 0) cur_feat = feat_[indices_[k] * c];

      out[i * c + j] += weight_[k] * cur_feat;
    }
  }
}

// input weight (N, 8), indices (N, 8), top_grad (N, c) -> bottom grad (n, c)
template <typename scalar_t>
__global__ void devoxelize_backward_kernel(
    int N, int n, int c, const int *__restrict__ indices,
    const scalar_t *__restrict__ weight, const scalar_t *__restrict__ top_grad,
    scalar_t *__restrict__ bottom_grad) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int i = index / c;
  int j = index % c;

  if (i < N) {
    const int *indices_ = indices + 8 * i;
    const scalar_t *weight_ = weight + 8 * i;

    scalar_t cur_top_grad = top_grad[i * c + j];

#pragma unroll
    for (int k = 0; k < 8; k++) {
      if (indices_[k] >= 0)
        atomicAdd(&bottom_grad[indices_[k] * c + j], weight_[k] * cur_top_grad);
    }
  }
}

// make sure indices is int type
// feat: (b,c,s) indices: (N, 3) batch_index: (N, ) -> out: (N, c)
at::Tensor devoxelize_forward_cuda(const at::Tensor feat,
                                   const at::Tensor indices,
                                   const at::Tensor weight) {
  int c = feat.size(1);
  int N = indices.size(0);

  at::Tensor out =
      torch::zeros({N, c}, at::device(feat.device()).dtype(feat.dtype()));

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      feat.type(), "devoxelize_forward_cuda", ([&] {
        devoxelize_forward_kernel<scalar_t><<<N, c>>>(
            N, c, indices.data_ptr<int>(), weight.data_ptr<scalar_t>(),
            feat.data_ptr<scalar_t>(), out.data_ptr<scalar_t>());
      }));

  return out;
}

// top_grad: (N, c), indices: (N, 3), batch_index: (N, ) -> bottom_grad:
// (b,c,s), s=r^3
at::Tensor devoxelize_backward_cuda(const at::Tensor top_grad,
                                    const at::Tensor indices,
                                    const at::Tensor weight, int n) {
  int c = top_grad.size(1);
  int N = top_grad.size(0);
  at::Tensor bottom_grad = torch::zeros(
      {n, c}, at::device(top_grad.device()).dtype(top_grad.dtype()));

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.type(), "devoxelize_backward_cuda", ([&] {
        devoxelize_backward_kernel<scalar_t><<<N, c>>>(
            N, n, c, indices.data_ptr<int>(), weight.data_ptr<scalar_t>(),
            top_grad.data_ptr<scalar_t>(), bottom_grad.data_ptr<scalar_t>());
      }));

  return bottom_grad;
}

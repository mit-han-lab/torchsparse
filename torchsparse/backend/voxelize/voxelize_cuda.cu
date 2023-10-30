#include <stdio.h>
#include <stdlib.h>
#include <torch/torch.h>

#include <THC/THCAtomics.cuh>
#include <cmath>

// to_dense: feats (N x C), coords (N x 4), output (B x H x W x D x C)
// coords: batch, x, y, z
template <typename scalar_t>
__global__ void to_dense_forward_kernel(int N, int c, const scalar_t *__restrict__ feats, const int *__restrict__ coords, const int *__restrict__ range, scalar_t *__restrict__ out)
{
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  int i = index / c;
  int j = index % c;
  if (i < N)
  {
    const int *cur_coords = coords + 4 * i;
    int pos = cur_coords[0] * range[1] * range[2] * range[3] + cur_coords[1] * range[2] * range[3] + cur_coords[2] * range[3] + cur_coords[3];
    out[pos * c + j] = feats[index];
  }
}

// to_dense: top_grad (B x H x W x D x C), coords (N x 4), bottom_grad (N x C)
template <typename scalar_t>
__global__ void to_dense_backward_kernel(int N, int c, const scalar_t *__restrict__ top_grad, const int *__restrict__ coords, const int *__restrict__ range, scalar_t *__restrict__ bottom_grad)
{
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  int i = index / c;
  int j = index % c;
  if (i < N)
  {
    const int *cur_coords = coords + 4 * i;
    int pos = cur_coords[0] * range[1] * range[2] * range[3] + cur_coords[1] * range[2] * range[3] + cur_coords[2] * range[3] + cur_coords[3];
    bottom_grad[index] = top_grad[pos * c + j];
  }
}

template <typename scalar_t>
__global__ void voxelize_forward_kernel(int N, int c, int s,
                                        const scalar_t *__restrict__ data,
                                        const int *__restrict__ idx,
                                        const int *__restrict__ counts,
                                        scalar_t *__restrict__ out)
{
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  int i = index / c;
  int j = index % c;
  if (i < N)
  {
    int pos = idx[i];
    if (pos < 0 || pos >= s || counts[pos] == 0)
      return;
    atomicAdd(&out[pos * c + j], data[i * c + j] / float(counts[pos]));
  }
}

template <typename scalar_t>
__global__ void voxelize_backward_kernel(int N, int c, int s,
                                         const scalar_t *__restrict__ top_grad,
                                         const int *__restrict__ idx,
                                         const int *__restrict__ counts,
                                         scalar_t *__restrict__ bottom_grad)
{
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  int i = index / c;
  int j = index % c;
  if (i < N)
  {
    int pos = idx[i];
    if (pos < 0 || pos >= s || counts[pos] == 0)
      return;
    atomicAdd(&bottom_grad[i * c + j],
              top_grad[pos * c + j] / float(counts[pos]));
  }
}

at::Tensor voxelize_forward_cuda(const at::Tensor inputs, const at::Tensor idx,
                                 const at::Tensor counts)
{
  int N = inputs.size(0);
  int c = inputs.size(1);
  int N1 = counts.size(0);

  at::Tensor out =
      torch::zeros({N1, c}, at::device(idx.device()).dtype(inputs.dtype()));

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      inputs.type(), "voxelize_forward_cuda", ([&]
                                               { voxelize_forward_kernel<scalar_t><<<N, c>>>(
                                                     N, c, N1, inputs.data_ptr<scalar_t>(), idx.data_ptr<int>(),
                                                     counts.data_ptr<int>(), out.data_ptr<scalar_t>()); }));

  return out;
}

at::Tensor voxelize_backward_cuda(const at::Tensor top_grad,
                                  const at::Tensor idx, const at::Tensor counts,
                                  const int N)
{
  int c = top_grad.size(1);
  int N1 = counts.size(0);

  at::Tensor bottom_grad =
      torch::zeros({N, c}, at::device(idx.device()).dtype(top_grad.dtype()));

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.type(), "voxelize_backward_cuda", ([&]
                                                  { voxelize_backward_kernel<scalar_t><<<N, c>>>(
                                                        N, c, N1, top_grad.data_ptr<scalar_t>(), idx.data_ptr<int>(),
                                                        counts.data_ptr<int>(), bottom_grad.data_ptr<scalar_t>()); }));

  return bottom_grad;
}

void to_dense_forward_cuda(const at::Tensor inputs, const at::Tensor idx,
                           const at::Tensor range, at::Tensor outputs)
{
  int N = inputs.size(0);
  int c = inputs.size(1);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      inputs.type(), "to_dense_forward_cuda", ([&]
                                               { to_dense_forward_kernel<scalar_t><<<(N * c + 255) / 256, 256>>>(
                                                     N, c, inputs.data_ptr<scalar_t>(), idx.data_ptr<int>(),
                                                     range.data_ptr<int>(), outputs.data_ptr<scalar_t>()); }));
}

void to_dense_backward_cuda(const at::Tensor top_grad,
                            const at::Tensor idx, const at::Tensor range,
                            const at::Tensor bottom_grad)
{
  int N = bottom_grad.size(0);
  int c = bottom_grad.size(1);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.type(), "to_dense_backward_cuda", ([&]
                                                  { to_dense_backward_kernel<scalar_t><<<(N * c + 255) / 256, 256>>>(
                                                        N, c, top_grad.data_ptr<scalar_t>(), idx.data_ptr<int>(),
                                                        range.data_ptr<int>(), bottom_grad.data_ptr<scalar_t>()); }));
}

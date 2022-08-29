#include <torch/extension.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdio>
#include <vector>

#include "downsample_cuda.h"
#define NDim 4
#define Tuple std::vector<int>

// take care of long.
// what about the batch index

/*
Transform input coordinates to 1D ravel hash values.
in_coords: N x (NDIM + 1)
coords_min: NDIM
coords_max: NDIM
out_coords: N
*/
__host__ __device__ inline int64_t transform_coords(int *in_coords,
                                                    int *coords_min,
                                                    int *coords_max) {
  int64_t cur = 0;
  int sizes[NDim];
#pragma unroll
  for (int i = 0; i < NDim; i++) sizes[i] = coords_max[i] - coords_min[i] + 1;
#pragma unroll
  for (int i = 0; i < NDim; i++) {
    cur *= sizes[i];
    cur += (in_coords[i] - coords_min[i]);
  }
  return cur;
}

/*
Transform 1D ravel hash values back to 3D coordinates.
in_coords: N
coords_min: NDIM
coords_max: NDIM
out_coords: N x (NDIM + 1)
*/
__host__ __device__ inline void inverse_transform_coords(int64_t *in_coords,
                                                         int *coords_min,
                                                         int *coords_max,
                                                         int *out_coords) {
  int cur = in_coords[0];
  int sizes[NDim];
#pragma unroll
  for (int i = 0; i < NDim; i++) sizes[i] = coords_max[i] - coords_min[i] + 1;
#pragma unroll
  for (int i = NDim - 1; i >= 0; i--) {
    out_coords[i] = coords_min[i] + (cur % sizes[i]);
    cur /= sizes[i];
  }
}

// note: in_coords point to current point. This is not a __global__ kernel!
// note: no need to store out_coords here, we'll restore them after unique_
// later! coords Nx4 or Nx3? impl. problematic
__host__ __device__ int get_output_coords(int kernel_volume, int *in_coords,
                                          int *kernel_sizes, int *stride,
                                          int *tensor_stride, int *coords_min,
                                          int *coords_max, int *out_coords,
                                          int64_t *transformed_coords) {
  int point_counter = 0;
  int upper[NDim - 1], lower[NDim - 1], counter[NDim - 1], cur;
  bool valid = false;
#pragma unroll
  for (int i = 0; i < NDim - 1; ++i) {
    counter[i] = 0;
    if (kernel_sizes[i] % 2 == 0) {
      lower[i] = 0;
      upper[i] = (kernel_sizes[i] - 1);
    } else {
      lower[i] = -(kernel_sizes[i] - 1) / 2;
      upper[i] = (kernel_sizes[i] - 1) / 2;
    }
  }
  for (int i = 0; i < kernel_volume; i++) {
    valid = true;
#pragma unroll
    // batch index no need to mod
    for (int j = NDim - 1; j >= 1; --j) {
      cur =
          in_coords[j] + (lower[j - 1] + counter[j - 1]) * tensor_stride[j - 1];
      if (cur % (tensor_stride[j - 1] * stride[j - 1]) == 0 &&
          cur >= coords_min[j]  //&&
                                // cur < coords_max[j]
      ) {
        out_coords[point_counter * NDim + j] = cur;
      } else
        valid = false;
    }
    counter[NDim - 2]++;
    out_coords[point_counter * NDim] = in_coords[0];

    if (valid) {
      point_counter++;
    }
#pragma unroll
    for (int c = NDim - 2; c >= 0; --c) {
      if (counter[c] == upper[c] - lower[c] + 1 && c > 0) {
        counter[c - 1]++;
        counter[c] = 0;
      }
    }
  }

  return point_counter;
}

__global__ void get_output_coords_kernel(int n_points, int n_kernel,
                                         int *in_coords, int *kernel_sizes,
                                         int *stride, int *tensor_stride,
                                         int *coords_min, int *coords_max,
                                         int *n_out_points,
                                         int64_t *transformed_coords) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= n_points) return;
  int coords_out[256 * NDim];
  int point_counter = get_output_coords(
      n_kernel, in_coords + idx * NDim, kernel_sizes, stride, tensor_stride,
      coords_min, coords_max, coords_out, transformed_coords);
  for (int i = 0; i < point_counter; i++) {
    int old_idx = atomicAdd(n_out_points, 1);
    int64_t cur_transformed_coord =
        transform_coords(coords_out + i * NDim, coords_min, coords_max);
    transformed_coords[old_idx] = cur_transformed_coord;
  }
}

__global__ void inverse_transform_coords_kernel(int n_points,
                                                int64_t *in_coords,
                                                int *coords_min,
                                                int *coords_max,
                                                int *out_coords) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_points) return;
  inverse_transform_coords(in_coords + idx, coords_min, coords_max,
                           out_coords + idx * NDim);
}

/*
Idea: launch get_output_coords_kernel then inverse_transform_coords_kernel

*/
at::Tensor downsample_cuda(at::Tensor in_coords, at::Tensor coords_max,
                           at::Tensor coords_min, Tuple kernel_sizes_h,
                           Tuple stride_h, Tuple tensor_stride_h) {
  int N = in_coords.size(0);
  int kernel_volume = std::accumulate(
      kernel_sizes_h.begin(), kernel_sizes_h.end(), 1, std::multiplies<int>{});
  int *kernel_sizes, *stride, *tensor_stride;
  cudaMalloc((void **)&kernel_sizes, (NDim - 1) * sizeof(int));
  cudaMalloc((void **)&stride, (NDim - 1) * sizeof(int));
  cudaMalloc((void **)&tensor_stride, (NDim - 1) * sizeof(int));
  cudaMemcpy(kernel_sizes, kernel_sizes_h.data(), (NDim - 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(stride, stride_h.data(), (NDim - 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(tensor_stride, tensor_stride_h.data(), (NDim - 1) * sizeof(int),
             cudaMemcpyHostToDevice);

  at::Tensor out_coords_transformed;
  out_coords_transformed =
      torch::zeros({kernel_volume * N}, torch::TensorOptions()
                                            .dtype(at::ScalarType::Long)
                                            .device(in_coords.device()));
  at::Tensor n_out_points;
  n_out_points = torch::zeros({1}, torch::TensorOptions()
                                       .dtype(at::ScalarType::Int)
                                       .device(in_coords.device()));
  get_output_coords_kernel<<<int(ceil((double)N / 256)), 256>>>(
      N, kernel_volume, in_coords.data_ptr<int>(), kernel_sizes, stride,
      tensor_stride, coords_min.data_ptr<int>(), coords_max.data_ptr<int>(),
      n_out_points.data_ptr<int>(), out_coords_transformed.data_ptr<long>());

  int n_out_points_scalar = (int)n_out_points.item<int>();
  out_coords_transformed = std::get<0>(at::_unique(
      at::slice(out_coords_transformed, 0, 0, n_out_points_scalar)));

  int num_out_points = out_coords_transformed.size(0);
  at::Tensor out_coords =
      torch::zeros({num_out_points, NDim}, torch::TensorOptions()
                                               .dtype(at::ScalarType::Int)
                                               .device(in_coords.device()));

  inverse_transform_coords_kernel<<<int(ceil((double)num_out_points / 256)),
                                    256>>>(
      num_out_points, out_coords_transformed.data_ptr<long>(),
      coords_min.data_ptr<int>(), coords_max.data_ptr<int>(),
      out_coords.data_ptr<int>());
  return out_coords;
}

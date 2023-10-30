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
  int64_t cur = in_coords[0];
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
                                          int *coords_min,
                                          int *coords_max,
                                          int *padding, 
                                          int *out_coords) {
  int point_counter = 0;
  int upper[NDim - 1], lower[NDim - 1], counter[NDim - 1], cur;
  bool valid = false;
#pragma unroll
  for (int i = 0; i < NDim - 1; ++i) {
    counter[i] = 0;
    lower[i] = -(kernel_sizes[i] - 1);
    upper[i] = 0;
  }
  for (int i = 0; i < kernel_volume; i++) {
    valid = true;
#pragma unroll
    // batch index no need to mod
    for (int j = NDim - 1; j >= 1; --j) {
      // cur = in_coords[j] + (lower[j - 1] + counter[j - 1]) ;
      cur = in_coords[j] + (lower[j - 1] + counter[j - 1]) + padding[j - 1];
      int cur_div = cur / stride[j - 1];
      if (((cur % (stride[j - 1])) == 0) &&
          (cur_div >= coords_min[j])  &&
          (cur_div <= coords_max[j])
      ) {
        out_coords[point_counter * NDim + j] = cur_div;
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
                                         int *stride,
                                         int *coords_min, int *coords_max,
                                         int *padding,
                                         int *n_out_points,
                                         int64_t *transformed_coords) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_points) return;
  int coords_out[256 * NDim];
  int point_counter = get_output_coords(
      n_kernel, in_coords + idx * NDim, kernel_sizes, stride,
      coords_min, coords_max, padding, coords_out);
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

at::Tensor downsample_cuda(at::Tensor _in_coords, at::Tensor _coords_max,
                           at::Tensor _coords_min, at::Tensor _kernel_sizes,
                           at::Tensor _stride, at::Tensor _padding) {
  
  int N = _in_coords.size(0);
  int kernel_volume = (int)(torch::prod(_kernel_sizes).item<int>());
  int *in_coords = _in_coords.data_ptr<int>();
  int *coords_min = _coords_min.data_ptr<int>();
  int *coords_max = _coords_max.data_ptr<int>();
  int *kernel_sizes = _kernel_sizes.data_ptr<int>();
  int *stride = _stride.data_ptr<int>();
  int *padding = _padding.data_ptr<int>();

  at::Tensor _out_coords_transformed = torch::zeros({kernel_volume * N}, torch::TensorOptions()
                                            .dtype(at::ScalarType::Long)
                                            .device(_in_coords.device()));
  at::Tensor _n_out_points = torch::zeros({1}, torch::TensorOptions()
                                       .dtype(at::ScalarType::Int)
                                       .device(_in_coords.device()));
  
  int* n_out_points = _n_out_points.data_ptr<int>();

  get_output_coords_kernel<<<int(ceil((double)N / 256)), 256>>>(
      N, kernel_volume, in_coords, kernel_sizes, stride,
      coords_min, coords_max, padding,
      n_out_points, _out_coords_transformed.data_ptr<long>());

  int n_out_points_scalar = (int)_n_out_points.item<int>();

  _out_coords_transformed = std::get<0>(at::_unique(
      at::slice(_out_coords_transformed, 0, 0, n_out_points_scalar)));

  int num_out_points = _out_coords_transformed.size(0);
  at::Tensor _out_coords = torch::zeros({num_out_points, NDim}, torch::TensorOptions()
                                               .dtype(at::ScalarType::Int)
                                               .device(_in_coords.device()));
  int* out_coords = _out_coords.data_ptr<int>();

  inverse_transform_coords_kernel<<<int(ceil((double)num_out_points / 256)), 256>>>(
      num_out_points, _out_coords_transformed.data_ptr<long>(),
      coords_min, coords_max, out_coords);

  return _out_coords;
}
#include <torch/extension.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdio>
#include <vector>
#define NDim 4
#define MAX_KVOL 27

__host__ __device__ inline int32_t transform_coords(int *in_coords,
                                                    int *coords_min,
                                                    int *coords_max) {
  int32_t cur = 0;
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

__host__ __device__ inline void inverse_transform_coords(int32_t *in_coords,
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

__host__ __device__ int get_output_coords_downsample(
    int kernel_volume, int *in_coords, int *kernel_sizes, int *stride,
    int *tensor_stride, int *coords_min, int *coords_max, int *out_coords,
    int *offsets) {
  int point_counter = 0;
  int upper[NDim], lower[NDim], counter[NDim], cur;
  bool valid = false;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    counter[i] = 0;
    if (i == 0) lower[i] = upper[i] = 0;
    if (kernel_sizes[i - 1] % 2 == 0) {
      lower[i] = 0;
      upper[i] = (kernel_sizes[i - 1] - 1);
    } else {
      lower[i] = -(kernel_sizes[i - 1] - 1) / 2;
      upper[i] = (kernel_sizes[i - 1] - 1) / 2;
    }
  }
  for (int i = 0; i < kernel_volume; i++) {
    valid = true;
#pragma unroll
    // batch index no need to mod
    for (int j = NDim - 1; j >= 1; --j) {
      /// if(j > 0) cur = in_coords[j] + (lower[j] + counter[j]) *
      /// tensor_stride[j - 1];
      // else cur = in_coords[j];
      cur = in_coords[j] + (lower[j] + counter[j]) * tensor_stride[j - 1];
      // if((j == 0) ||
      if ((cur % (tensor_stride[j - 1] * stride[j - 1]) == 0 &&
           cur >= coords_min[j] && cur <= coords_max[j])) {
        out_coords[point_counter * NDim + j] = cur;
        offsets[point_counter] = i;
      } else
        valid = false;
    }
    counter[NDim - 1]++;
    out_coords[point_counter * NDim] = in_coords[0];

    if (valid) {
      point_counter++;
    }
#pragma unroll
    for (int c = NDim - 1; c >= 0; --c) {
      if (counter[c] == upper[c] - lower[c] + 1 && c > 0) {
        counter[c - 1]++;
        counter[c] = 0;
      }
    }
  }

  return point_counter;
}

__host__ __device__ int get_output_coords_subm(int kernel_volume,
                                               int *in_coords,
                                               int *kernel_sizes,
                                               int *coords_min, int *coords_max,
                                               int *out_coords, int *offsets) {
  int point_counter = 0;
  int upper[NDim], lower[NDim], counter[NDim], cur;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    counter[i] = 0;
    if (i == 0) lower[i] = upper[i] = 0;
    if (kernel_sizes[i - 1] % 2 == 0) {
      lower[i] = 0;
      upper[i] = (kernel_sizes[i - 1] - 1);
    } else {
      lower[i] = -(kernel_sizes[i - 1] - 1) / 2;
      upper[i] = (kernel_sizes[i - 1] - 1) / 2;
    }
  }
  bool valid;
  for (int i = 0; i < kernel_volume; i++) {
    valid = true;
#pragma unroll
    // batch index no need to mod
    for (int j = NDim - 1; j >= 1; --j) {
      cur = in_coords[j] + (lower[j] + counter[j]);
      if (cur >= coords_min[j] && cur <= coords_max[j]) {
        out_coords[point_counter * NDim + j] = cur;
        offsets[point_counter] = i;
      } else
        valid = false;
    }
    out_coords[point_counter * NDim] = in_coords[0];
    counter[NDim - 1]++;
    if (valid) {
      point_counter++;
    }
#pragma unroll
    for (int c = NDim - 1; c >= 0; --c) {
      if (counter[c] == upper[c] - lower[c] + 1 && c > 0) {
        counter[c - 1]++;
        counter[c] = 0;
      }
    }
  }

  return point_counter;
}

__global__ void inverse_transform_coords_kernel(int n_points,
                                                int32_t *in_coords,
                                                int *coords_min,
                                                int *coords_max,
                                                int *out_coords) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_points) return;
  inverse_transform_coords(in_coords + idx, coords_min, coords_max,
                           out_coords + idx * NDim);
}

// get (not-unique) output coords,
// record input idx and output coord in kernel map
__global__ void downsample_grid_kmap_stage1_specialized(
    int n_points, int n_kernel, int *in_coords, int *kernel_sizes, int *stride,
    int *tensor_stride, int *coords_min, int *coords_max, int *n_out_points,
    int32_t *transformed_coords, int32_t *out_kmap, int *kmap_sizes) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // a little bit more complicated. need to consider offset.
  if (idx >= n_points) return;
  int cnt = 0;
  int coords_out[NDim], coords_out_tmp[NDim];
  coords_out[0] = in_coords[idx * NDim];
  coords_out_tmp[0] = in_coords[idx * NDim];
#pragma unroll
  for (int i = -(kernel_sizes[0] - 1) / 2; i <= (kernel_sizes[0] - 1) / 2;
       i++) {
    coords_out[1] = in_coords[idx * NDim + 1] / tensor_stride[0] + i;
#pragma unroll
    for (int j = -(kernel_sizes[1] - 1) / 2; j <= (kernel_sizes[1] - 1) / 2;
         j++) {
      coords_out[2] = in_coords[idx * NDim + 2] / tensor_stride[1] + j;
#pragma unroll
      for (int k = -(kernel_sizes[2] - 1) / 2; k <= (kernel_sizes[2] - 1) / 2;
           k++) {
        coords_out[3] = in_coords[idx * NDim + 3] / tensor_stride[2] + k;
        if (coords_out[1] % stride[0] != 0 || coords_out[2] % stride[1] != 0 ||
            coords_out[3] % stride[2] != 0) {
          cnt++;
          continue;
        }
        // manual unroll this one
        coords_out_tmp[1] = coords_out[1] / stride[0];
        coords_out_tmp[2] = coords_out[2] / stride[1];
        coords_out_tmp[3] = coords_out[3] / stride[2];

        if (coords_out_tmp[1] >= coords_min[1] &&
            coords_out_tmp[1] <= coords_max[1] &&
            coords_out_tmp[2] >= coords_min[2] &&
            coords_out_tmp[2] <= coords_max[2] &&
            coords_out_tmp[3] >= coords_min[3] &&
            coords_out_tmp[3] <= coords_max[3]) {
          int32_t grid_index =
              transform_coords(coords_out_tmp, coords_min, coords_max);
          int old_num = atomicAdd(&kmap_sizes[cnt], 1);
          int old_idx = atomicAdd(n_out_points, 1);
          transformed_coords[old_idx] = grid_index;
          out_kmap[((cnt * n_points) << 1) + (old_num << 1)] = idx;
          out_kmap[((cnt * n_points) << 1) + (old_num << 1) + 1] = grid_index;
        }

        cnt++;
      }
    }
  }
}

// get (not-unique) output coords,
// record input idx and output coord in kernel map
__global__ void downsample_grid_kmap_stage1(
    int n_points, int n_kernel, int *in_coords, int *kernel_sizes, int *stride,
    int *tensor_stride, int *coords_min, int *coords_max, int *n_out_points,
    int32_t *transformed_coords, int32_t *out_kmap, int *kmap_sizes) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // a little bit more complicated. need to consider offset.
  if (idx >= n_points) return;
  int coords_out[MAX_KVOL * NDim], offsets[MAX_KVOL], local_coords_min[NDim],
      local_coords_max[NDim];
  int point_counter = get_output_coords_downsample(
      n_kernel, in_coords + idx * NDim, kernel_sizes, stride, tensor_stride,
      coords_min, coords_max, coords_out, offsets);
  local_coords_max[0] = coords_max[0];
  local_coords_min[0] = coords_min[0];
#pragma unroll
  for (int i = 0; i < NDim - 1; i++) {
    local_coords_max[i + 1] =
        coords_max[i + 1] / (stride[i] * tensor_stride[i]);
    local_coords_min[i + 1] =
        coords_min[i + 1] / (stride[i] * tensor_stride[i]);
  }
  for (int i = 0; i < point_counter; i++) {
// divide the coordinates by stride * tensor stride
// this is to save space for grids
#pragma unroll
    for (int j = 0; j < NDim - 1; j++) {
      coords_out[i * NDim + j + 1] /= stride[j] * tensor_stride[j];
    }
    // coords_out[point_counter * NDim] = in_coords[idx * NDim];
    int old_idx = atomicAdd(n_out_points, 1);
    int32_t cur_transformed_coord = transform_coords(
        coords_out + i * NDim, local_coords_min, local_coords_max);
    int offset = offsets[i];
    transformed_coords[old_idx] = cur_transformed_coord;
    // record the transformed coord in kmap temporarily, later replace it with
    // out idx.
    int old_num = atomicAdd(&kmap_sizes[offset], 1);

    out_kmap[((offset * n_points) << 1) + (old_num << 1)] = idx;
    out_kmap[((offset * n_points) << 1) + (old_num << 1) + 1] =
        cur_transformed_coord;
  }
}

// construct the grid map based on unique coords
__global__ void downsample_grid_kmap_stage2(int n_points, int32_t *in_coords,
                                            int *out_grids) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_points) return;
  out_grids[in_coords[idx]] = idx;
}

// replace the coords in the output map with the output idx
__global__ void downsample_grid_kmap_stage3(int n_points, int n_points_out,
                                            int kernel_volume, int *grids,
                                            int32_t *in_kmap, int *kmap_sizes,
                                            int *cum_kmap_sizes, int *out_kmap,
                                            int *input_mask, int *output_mask) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_points) return;
  for (int i = 0; i < kernel_volume; i++) {
    int kmap_size = kmap_sizes[i];
    int cum_size = i == 0 ? 0 : cum_kmap_sizes[i - 1];
    int32_t *cur_in_kmap = in_kmap + 2 * n_points * i;
    if (idx >= kmap_size) continue;
    // manual unroll
    int input_idx = (int)cur_in_kmap[idx * 2];
    int output_idx = (int)grids[cur_in_kmap[idx * 2 + 1]];
    // temporary, to be removed.
    out_kmap[(cum_size + idx) * 2] = input_idx;
    out_kmap[(cum_size + idx) * 2 + 1] = output_idx;

    /*
    input_mask[input_idx * kernel_volume + i] = idx;
    output_mask[output_idx * kernel_volume + i] = idx;
    */

    // another layout
    input_mask[i * n_points + input_idx] = idx;
    output_mask[i * n_points_out + output_idx] = idx;
  }
}

__global__ void subm_grid_kmap_stage1(int n_points, int kernel_volume,
                                      int *in_coords, int *coords_min,
                                      int *coords_max, int *out_grids) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_points) return;
  int32_t grid_index =
      transform_coords(in_coords + idx * NDim, coords_min, coords_max);
  out_grids[grid_index] = idx;
}

__global__ void subm_grid_kmap_stage2(int n_points, int kernel_volume,
                                      int *in_coords, int *coords_min,
                                      int *coords_max, int *kernel_sizes,
                                      int *stride, int *tensor_stride,
                                      int *grids, int *out_kmap,
                                      int *kmap_sizes) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_points) return;
  /*
  int coords_out[MAX_KVOL * NDim], offsets[MAX_KVOL];

  int n_neighbors = get_output_coords_subm(
    kernel_volume, in_coords + idx * NDim, kernel_sizes,
    coords_min, coords_max, coords_out, offsets
  );
#pragma unroll
  for(int i = 0; i < n_neighbors / 2; i++){
    int32_t grid_index = transform_coords(coords_out + i * NDim, coords_min,
coords_max); int input_idx = grids[grid_index]; if(input_idx != -1){ int old_num
= atomicAdd(&kmap_sizes[i], 1); atomicAdd(&kmap_sizes[kernel_volume-1-i], 1);
      // kernel_volume x num_points x 2
      out_kmap[((i * n_points) << 1) + (old_num << 1)] = input_idx;
      out_kmap[((i * n_points) << 1) + (old_num << 1) + 1] = idx;
    }
  }
  int old_num = atomicAdd(&kmap_sizes[n_neighbors / 2], 1);
  //out_kmap[((i * n_points) << 1) + (old_num << 1)] = idx;
  //out_kmap[((i * n_points) << 1) + (old_num << 1) + 1] = idx;
  */
  int cnt = 0;
  int coords_out[NDim];
  coords_out[0] = in_coords[idx * NDim];
#pragma unroll
  for (int i = -(kernel_sizes[0] - 1) / 2; i <= (kernel_sizes[0] - 1) / 2;
       i++) {
    coords_out[1] = in_coords[idx * NDim + 1] + i;
#pragma unroll
    for (int j = -(kernel_sizes[1] - 1) / 2; j <= (kernel_sizes[1] - 1) / 2;
         j++) {
      coords_out[2] = in_coords[idx * NDim + 2] + j;
#pragma unroll
      for (int k = -(kernel_sizes[2] - 1) / 2; k <= (kernel_sizes[2] - 1) / 2;
           k++) {
        if (cnt == kernel_volume / 2) {
          out_kmap[((cnt * n_points) << 1) + (idx << 1)] = idx;
          out_kmap[((cnt * n_points) << 1) + (idx << 1) + 1] = idx;
          atomicAdd(&kmap_sizes[cnt], 1);
        } else if (cnt > kernel_volume / 2)
          continue;
        else {
          coords_out[3] = in_coords[idx * NDim + 3] + k;
          // manual unroll this one
          if (coords_out[1] >= coords_min[1] &&
              coords_out[1] <= coords_max[1] &&
              coords_out[2] >= coords_min[2] &&
              coords_out[2] <= coords_max[2] &&
              coords_out[3] >= coords_min[3] &&
              coords_out[3] <= coords_max[3]) {
            int32_t grid_index =
                transform_coords(coords_out, coords_min, coords_max);

            int input_idx = grids[grid_index];
            if (input_idx != -1) {
              int old_num = atomicAdd(&kmap_sizes[cnt], 1);
              atomicAdd(&kmap_sizes[kernel_volume - 1 - cnt], 1);
              out_kmap[((cnt * n_points) << 1) + (old_num << 1)] = input_idx;
              out_kmap[((cnt * n_points) << 1) + (old_num << 1) + 1] = idx;
              out_kmap[(((kernel_volume - 1 - cnt) * n_points) << 1) +
                       (old_num << 1)] = idx;
              out_kmap[(((kernel_volume - 1 - cnt) * n_points) << 1) +
                       (old_num << 1) + 1] = input_idx;
              // input_mask[input_idx * kernel_volume + cnt] = old_num;
              // output_mask[idx * kernel_volume + cnt] = old_num;
              // input_mask[idx * kernel_volume + kernel_volume-1-cnt] =
              // old_num; output_mask[input_idx * kernel_volume +
              // kernel_volume-1-cnt] = old_num;
            }
          }
        }
        ++cnt;
      }
    }
  }
}

__global__ void subm_grid_kmap_stage3(int n_points, int kernel_volume,
                                      int *in_kmap, int *kmap_sizes,
                                      int *cum_kmap_sizes, int *out_kmap,
                                      int *input_mask, int *output_mask) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_points) return;
  for (int i = 0; i < kernel_volume; i++) {
    int kmap_size = kmap_sizes[i];
    int cum_size = (i == 0) ? 0 : cum_kmap_sizes[i - 1];
    int *cur_in_kmap = in_kmap + 2 * n_points * i;
    if (idx >= kmap_size) continue;
    // manual unroll
    int input_idx = (int)cur_in_kmap[idx * 2];
    int output_idx = (int)cur_in_kmap[idx * 2 + 1];
    out_kmap[(cum_size + idx) * 2] = cur_in_kmap[idx * 2];
    out_kmap[(cum_size + idx) * 2 + 1] = cur_in_kmap[idx * 2 + 1];
    /*
    input_mask[input_idx * kernel_volume + i] = idx;
    // save half of the irregular write because
    // output_mask[output_idx * kernel_volume + i] = input_mask[input_idx *
    kernel_volume + i] output_mask[output_idx * kernel_volume + i] = idx;
    */
    // another layout
    input_mask[i * n_points + input_idx] = idx;
    output_mask[i * n_points + output_idx] = idx;

    // int cum_size2 = cum_kmap_sizes[kernel_volume - 1 - i - 1];
    // int *cur_in_kmap2 = in_kmap + 2 * n_points * (kernel_volume - 1 -i);
    // out_kmap[(cum_size2 + idx) * 2] = cur_in_kmap2[idx * 2];
    // out_kmap[(cum_size2 + idx) * 2 + 1] = cur_in_kmap2[idx * 2 + 1];
  }
}

// replace the coords in the output map with the output idx
__global__ void get_masks_from_kmap_kernel(int n_points, int n_points_out,
                                           int kernel_volume, int *kmap,
                                           int *kmap_sizes, int *cum_kmap_sizes,
                                           int *input_mask, int *output_mask) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_points) return;
  for (int i = 0; i < kernel_volume; i++) {
    if (n_points == n_points_out && kernel_volume % 2 == 1 &&
        i == kernel_volume / 2)
      continue;
    int kmap_size = kmap_sizes[i];
    int cum_size = i == 0 ? 0 : cum_kmap_sizes[i - 1];
    int *cur_in_kmap = kmap + cum_size * 2;
    if (idx >= kmap_size) continue;
    // manual unroll
    int input_idx = cur_in_kmap[idx * 2];
    int output_idx = cur_in_kmap[idx * 2 + 1];
    // another layout
    input_mask[i * n_points + input_idx] = idx;
    output_mask[i * n_points_out + output_idx] = idx;
  }
}

// Impl. trick: To save space, the grid index should / tensor_stride.

std::vector<at::Tensor> build_kernel_map_subm(
    at::Tensor _in_coords, at::Tensor _coords_min, at::Tensor _coords_max,
    at::Tensor _kernel_sizes, at::Tensor _stride, at::Tensor _tensor_stride) {
  int n_points = _in_coords.size(0);
  int kernel_volume = (int)(torch::prod(_kernel_sizes).item<int>());
  at::Tensor augmented_tensor_stride =
      torch::cat({torch::ones_like(_tensor_stride.index(
                      {torch::indexing::Slice({torch::indexing::None, 1})})),
                  _tensor_stride},
                 0);
  at::Tensor _in_coords_tmp =
      (_in_coords / augmented_tensor_stride).to(at::ScalarType::Int);
  at::Tensor _coords_min_tmp =
      (_coords_min / augmented_tensor_stride).to(at::ScalarType::Int);
  at::Tensor _coords_max_tmp =
      (_coords_max / augmented_tensor_stride).to(at::ScalarType::Int);
  int *in_coords = _in_coords_tmp.data_ptr<int>();
  int *coords_min = _coords_min_tmp.data_ptr<int>();
  int *coords_max = _coords_max_tmp.data_ptr<int>();
  int *kernel_sizes = _kernel_sizes.data_ptr<int>();
  int *stride = _stride.data_ptr<int>();
  int *tensor_stride = _tensor_stride.data_ptr<int>();
  auto options = torch::TensorOptions()
                     .dtype(at::ScalarType::Int)
                     .device(_in_coords.device());
  // auto options_long =
  // torch::TensorOptions().dtype(at::ScalarType::Long).device(_in_coords.device());
  at::Tensor _grids = torch::full(
      {(int)(torch::prod((_coords_max - _coords_min) / augmented_tensor_stride +
                         1)
                 .item<int>())},
      -1, options);
  int *grids = _grids.data_ptr<int>();
  at::Tensor _out_kmap =
      torch::full({kernel_volume * n_points * 2}, -1, options);
  at::Tensor _kmap_sizes = torch::zeros({kernel_volume}, options);
  at::Tensor _input_mask = torch::full({kernel_volume * n_points}, -1, options);
  at::Tensor _output_mask =
      torch::full({kernel_volume * n_points}, -1, options);
  int *out_kmap = _out_kmap.data_ptr<int>();
  int *kmap_sizes = _kmap_sizes.data_ptr<int>();
  int *input_mask = _input_mask.data_ptr<int>();
  int *output_mask = _output_mask.data_ptr<int>();
  // stage1: insert to grid
  subm_grid_kmap_stage1<<<(int)ceil((double)n_points / 256), 256>>>(
      n_points, kernel_volume, in_coords, coords_min, coords_max, grids);
  // stage2: query
  subm_grid_kmap_stage2<<<(int)ceil((double)n_points / 256), 256>>>(
      n_points, kernel_volume, in_coords, coords_min, coords_max, kernel_sizes,
      stride, tensor_stride, grids, out_kmap, kmap_sizes);
  // stage3: filter out all -1s in the kmap;
  at::Tensor final_out_kmap =
      torch::zeros({(int)(torch::sum(_kmap_sizes).item<int>()), 2}, options);
  at::Tensor _cum_kmap_sizes =
      torch::cumsum(_kmap_sizes, 0).to(at::ScalarType::Int);
  int *cum_kmap_sizes = _cum_kmap_sizes.data_ptr<int>();
  subm_grid_kmap_stage3<<<
      (int)ceil((double)(torch::max(_cum_kmap_sizes).item<int>()) / 256),
      256>>>(n_points, kernel_volume, out_kmap, kmap_sizes, cum_kmap_sizes,
             final_out_kmap.data_ptr<int>(), input_mask, output_mask);
  return {final_out_kmap, _kmap_sizes, _input_mask, _output_mask};
}

std::vector<at::Tensor> build_kernel_map_downsample(
    at::Tensor _in_coords, at::Tensor _coords_min, at::Tensor _coords_max,
    at::Tensor _kernel_sizes, at::Tensor _stride, at::Tensor _tensor_stride) {
  int n_points = _in_coords.size(0);
  int kernel_volume = (int)(torch::prod(_kernel_sizes).item<int>());
  int *in_coords = _in_coords.data_ptr<int>();
  int *coords_min = _coords_min.data_ptr<int>();
  int *coords_max = _coords_max.data_ptr<int>();
  int *kernel_sizes = _kernel_sizes.data_ptr<int>();
  int *stride = _stride.data_ptr<int>();
  int *tensor_stride = _tensor_stride.data_ptr<int>();
  auto options = torch::TensorOptions()
                     .dtype(at::ScalarType::Int)
                     .device(_in_coords.device());
  auto options_long = torch::TensorOptions()
                          .dtype(at::ScalarType::Int)
                          .device(_in_coords.device());

  at::Tensor _out_kmap =
      torch::zeros({kernel_volume * n_points * 2}, options_long) - 1;
  at::Tensor _kmap_sizes = torch::zeros({kernel_volume}, options);
  at::Tensor _n_out_points = torch::zeros({1}, options);
  at::Tensor _transformed_out_coords =
      torch::zeros({kernel_volume * n_points}, options_long);
  // transformed coordinates is long
  int32_t *out_kmap = _out_kmap.data_ptr<int>();
  int *kmap_sizes = _kmap_sizes.data_ptr<int>();
  int *n_out_points = _n_out_points.data_ptr<int>();
  int32_t *transformed_out_coords = _transformed_out_coords.data_ptr<int>();

  /*
  // If we do the general downsample stage 1, we do it here as follows:
  */

  /*
  // stage1: calculate out_coordinates and setup the initial kernel map
  downsample_grid_kmap_stage1<<<(int)ceil((double)n_points / 256),
  256>>>(n_points, kernel_volume, in_coords, kernel_sizes, stride,
  tensor_stride, coords_min, coords_max, n_out_points, transformed_out_coords,
  out_kmap, kmap_sizes);

  */

  // after stage 1, we can alter coords_min, coords_max, etc.
  at::Tensor augmented_tensor_stride =
      torch::cat({torch::ones_like(_tensor_stride.index(
                      {torch::indexing::Slice({torch::indexing::None, 1})})),
                  _tensor_stride * _stride},
                 0);
  at::Tensor _coords_min_tmp =
      (_coords_min / augmented_tensor_stride).to(at::ScalarType::Int);
  at::Tensor _coords_max_tmp =
      (_coords_max / augmented_tensor_stride).to(at::ScalarType::Int);
  coords_min = _coords_min_tmp.data_ptr<int>();
  coords_max = _coords_max_tmp.data_ptr<int>();

  /*
  // If we do specialized downsample for 3D coords (stage 1), we do it (using
  divided coords_min/max) as follows:
  */
  downsample_grid_kmap_stage1_specialized<<<(int)ceil((double)n_points / 256),
                                            256>>>(
      n_points, kernel_volume, in_coords, kernel_sizes, stride, tensor_stride,
      coords_min, coords_max, n_out_points, transformed_out_coords, out_kmap,
      kmap_sizes);

  at::Tensor _grids = torch::full(
      {(int)(torch::prod(_coords_max_tmp - _coords_min_tmp + 1).item<int>())},
      -1, options);
  int *grids = _grids.data_ptr<int>();
  // stage2: get unique coordinates and insert them to the grid.
  at::Tensor _out_coords = std::get<0>(torch::_unique(_transformed_out_coords));
  int32_t *out_coords = _out_coords.data_ptr<int>();
  // stage 2.1: transform the out coords to N x 4 format.
  int n_out_points_scalar = _out_coords.size(0);
  at::Tensor final_out_coords =
      torch::zeros({n_out_points_scalar, NDim}, options);
  inverse_transform_coords_kernel<<<
      (int)ceil((double)n_out_points_scalar / 256), 256>>>(
      n_out_points_scalar, out_coords, coords_min, coords_max,
      final_out_coords.data_ptr<int>());
  // restore coords
  final_out_coords *= augmented_tensor_stride;

  downsample_grid_kmap_stage2<<<(int)ceil((double)n_out_points_scalar / 256),
                                256>>>(n_out_points_scalar, out_coords, grids);
  // stage3: replace the (64b) coordinate ravel hashes with the output idx
  at::Tensor final_out_kmap =
      torch::zeros({(int)(torch::sum(_kmap_sizes).item<int>()), 2}, options);
  at::Tensor _cum_kmap_sizes =
      torch::cumsum(_kmap_sizes, 0).to(at::ScalarType::Int);

  at::Tensor _input_mask = torch::full({kernel_volume * n_points}, -1, options);
  at::Tensor _output_mask =
      torch::full({kernel_volume * n_out_points_scalar}, -1, options);
  int *input_mask = _input_mask.data_ptr<int>();
  int *output_mask = _output_mask.data_ptr<int>();

  int *cum_kmap_sizes = _cum_kmap_sizes.data_ptr<int>();
  downsample_grid_kmap_stage3<<<
      (int)ceil((double)torch::max(_cum_kmap_sizes).item<int>() / 256), 256>>>(
      n_points, n_out_points_scalar, kernel_volume, grids, out_kmap, kmap_sizes,
      cum_kmap_sizes, final_out_kmap.data_ptr<int>(), input_mask, output_mask);
  return {final_out_kmap, _kmap_sizes, final_out_coords, _input_mask,
          _output_mask};
}

std::vector<at::Tensor> build_mask_from_kmap(int n_points, int n_out_points,
                                             at::Tensor _kmap,
                                             at::Tensor _kmap_sizes) {
  int kernel_volume = _kmap_sizes.size(0);
  auto options =
      torch::TensorOptions().dtype(at::ScalarType::Int).device(_kmap.device());
  at::Tensor _kmap_sizes_cpu = _kmap_sizes.to(torch::kCPU);
  at::Tensor _cum_kmap_sizes =
      torch::cumsum(_kmap_sizes, 0).to(at::ScalarType::Int);
  at::Tensor _input_mask = torch::full({kernel_volume * n_points}, -1, options);
  at::Tensor _output_mask =
      torch::full({kernel_volume * n_out_points}, -1, options);
  int *kmap = _kmap.data_ptr<int>();
  int *kmap_sizes = _kmap_sizes.data_ptr<int>();
  int *cum_kmap_sizes = _cum_kmap_sizes.data_ptr<int>();
  int *input_mask = _input_mask.data_ptr<int>();
  int *output_mask = _output_mask.data_ptr<int>();

  int max_kmap_size = 1;
  if (kernel_volume % 2 == 1 && n_points == n_out_points) {
    max_kmap_size =
        *std::max_element(_kmap_sizes_cpu.data_ptr<int>(),
                          _kmap_sizes_cpu.data_ptr<int>() + kernel_volume / 2);
    max_kmap_size =
        std::max(max_kmap_size,
                 *std::max_element(
                     _kmap_sizes_cpu.data_ptr<int>() + kernel_volume / 2 + 1,
                     _kmap_sizes_cpu.data_ptr<int>() + kernel_volume));
    max_kmap_size = std::max(max_kmap_size, 1);
  } else {
    max_kmap_size =
        *std::max_element(_kmap_sizes_cpu.data_ptr<int>(),
                          _kmap_sizes_cpu.data_ptr<int>() + kernel_volume);
  }
  get_masks_from_kmap_kernel<<<ceil((double)max_kmap_size / 256), 256>>>(
      n_points, n_out_points, kernel_volume, kmap, kmap_sizes, cum_kmap_sizes,
      input_mask, output_mask);
  return {_input_mask, _output_mask};
}

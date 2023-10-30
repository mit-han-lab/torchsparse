#include <torch/extension.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdio>
#include <vector>
#include "../hashmap/hashmap_cuda.cuh"
#define NDim 4
#define MAX_KVOL 27


template <typename type_int>  // int32_t or int64_t
__host__ __device__ inline type_int transform_coords(int *in_coords,
                                                    int *coords_min,
                                                    int *coords_max) {
  type_int cur = 0;
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

template <typename type_int>  // int32_t or int64_t
__host__ __device__ inline void inverse_transform_coords(type_int *in_coords,
                                                         int *coords_min,
                                                         int *coords_max,
                                                         int *out_coords) {
  type_int cur = in_coords[0];
  int sizes[NDim];
#pragma unroll
  for (int i = 0; i < NDim; i++) sizes[i] = coords_max[i] - coords_min[i] + 1;
#pragma unroll
  for (int i = NDim - 1; i >= 0; i--) {
    out_coords[i] = coords_min[i] + (cur % sizes[i]);
    cur /= sizes[i];
  }
}

template <typename type_hashtable_device_view, typename type_int>  // int32_t or int64_t
__global__ void inverse_transform_coords_and_insert_kernel(
                                                type_hashtable_device_view table,
                                                int n_points,
                                                type_int *in_coords,
                                                int *coords_min,
                                                int *coords_max,
                                                int *out_coords) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_points) return;
  inverse_transform_coords(in_coords + idx, coords_min, coords_max,
                           out_coords + idx * NDim);
  table.insert(in_coords[idx] + 1, idx + 1);
}


template <typename type_int, bool odd>  // int32_t or int64_t
__global__ void downsample_grid_kmap_stage1_specialized_fast(
    int n_points, int kernel_volume, int *in_coords, int *kernel_sizes, int *stride,
    int *padding, int *coords_min, int *coords_max, int *n_out_points,
    type_int *transformed_coords, type_int *out_in_map) {
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = tidx / kernel_volume;
  int _kernel_idx = tidx % kernel_volume;
  int kernel_idx = _kernel_idx;
  if (idx >= n_points) return;
  int coords_out[NDim];
  coords_out[0] = in_coords[idx * NDim];  //batch_idx
  if constexpr (odd)
  {
    #pragma unroll
    for(int i = 1; i <= NDim - 1; i++){
      int cur_offset = _kernel_idx % kernel_sizes[i - 1];
      cur_offset -= (kernel_sizes[i - 1] - 1);
      coords_out[i] = in_coords[idx * NDim + i] + padding[i - 1] + cur_offset;
      if(coords_out[i] % stride[i - 1] != 0) return;
      coords_out[i] /= stride[i - 1]; 
      _kernel_idx /= kernel_sizes[i - 1];
    }
  }
  else
  {
    #pragma unroll
    for(int i = NDim - 1; i >= 1; i--){
      int cur_offset = _kernel_idx % kernel_sizes[i - 1];
      cur_offset -= (kernel_sizes[i - 1] - 1);
      coords_out[i] = in_coords[idx * NDim + i] + padding[i - 1] + cur_offset;
      if(coords_out[i] % stride[i - 1] != 0) return;
      coords_out[i] /= stride[i - 1];
      _kernel_idx /= kernel_sizes[i - 1];
    }
  }
  if (coords_out[1] >= coords_min[1] &&
    coords_out[1] <= coords_max[1] &&
    coords_out[2] >= coords_min[2] &&
    coords_out[2] <= coords_max[2] &&
    coords_out[3] >= coords_min[3] &&
    coords_out[3] <= coords_max[3]) {
    type_int grid_index = transform_coords<type_int>(coords_out, coords_min, coords_max);
    int old_idx = atomicAdd(n_out_points, 1);
    transformed_coords[old_idx] = grid_index;
    out_in_map[idx * kernel_volume + kernel_idx] = grid_index;
  }
}


template <typename type_hashtable_device_view, typename type_int>  // int32_t or int64_t
__global__ void downsample_hashmap_kmap_stage3(type_hashtable_device_view table,
                                            int n_points, int n_points_out,
                                            int kernel_volume, 
                                            type_int *in_out_in_map,
                                            int *out_in_map) {
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = tidx / kernel_volume;
  int kernel_idx = tidx % kernel_volume;
  if (idx >= n_points) return;
  int opt_coords = in_out_in_map[tidx];
  if(opt_coords >= 0){
    int oidx = table.lookup(opt_coords + 1) - 1;
    //if(oidx < 0 || oidx >= n_points_out) printf("%d %d\n", opt_coords, oidx);
    out_in_map[oidx * kernel_volume + kernel_volume - 1 - kernel_idx] = idx;
  }
}


template <typename type_hashtable_device_view, typename type_int>  // int32_t or int64_t
__global__ void subm_hashmap_kmap_stage1(type_hashtable_device_view table,
                                      int n_points, int kernel_volume,
                                      int *in_coords, int *coords_min,
                                      int *coords_max, type_int *out_coords) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_points) return;
  type_int grid_index =
      transform_coords<type_int>(in_coords + idx * NDim, coords_min, coords_max);   // 4D to 1D
  out_coords[idx] = grid_index;
  table.insert(grid_index + 1, idx + 1);
}


// only support odd kernel shapes
template <typename type_hashtable_device_view, typename type_int>  // int32_t or int64_t
__global__ void subm_hashmap_kmap_stage2_odd_kernel(type_hashtable_device_view table,
                                      int n_points, int kernel_volume,
                                      int *in_coords, int *coords_min,
                                      int *coords_max, int *kernel_sizes,
                                      int *out_in_map) {
  
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = tidx / (kernel_volume / 2);
  int _kernel_idx = tidx % (kernel_volume / 2);
  int kernel_idx = _kernel_idx;
  if (idx >= n_points) return;
   
  if (_kernel_idx == 0){
    out_in_map[idx * kernel_volume + kernel_volume / 2] = idx;
  }

  int coords_out[NDim];
  coords_out[0] = in_coords[idx * NDim];
  
  #pragma unroll
  for(int i = 1; i <= NDim - 1; i++){
    int cur_offset = _kernel_idx % kernel_sizes[i - 1];
    cur_offset -= (kernel_sizes[i - 1] - 1) / 2;              
    coords_out[i] = in_coords[idx * NDim + i] + cur_offset;
    _kernel_idx /= kernel_sizes[i - 1];
  }
  
  
  if (coords_out[1] >= coords_min[1] &&
    coords_out[1] <= coords_max[1] &&
    coords_out[2] >= coords_min[2] &&
    coords_out[2] <= coords_max[2] &&
    coords_out[3] >= coords_min[3] &&
    coords_out[3] <= coords_max[3]) {
  
    type_int grid_index = transform_coords<type_int>(coords_out, coords_min, coords_max);
    int input_idx = table.lookup(grid_index + 1) - 1;
    if (input_idx >= 0) {
      out_in_map[idx * kernel_volume + kernel_idx] = input_idx;
      out_in_map[input_idx * kernel_volume + kernel_volume - 1 - kernel_idx] = idx;
    }
  }
  
}


// support even kernel shapes
template <typename type_hashtable_device_view, typename type_int>  // int32_t or int64_t
__global__ void subm_hashmap_kmap_stage2_even_kernel(type_hashtable_device_view table,
                                                      int n_points, int kernel_volume,
                                                      int *in_coords, int *coords_min,
                                                      int *coords_max, int *kernel_sizes,
                                                      int *out_in_map) {
                  
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = tidx / kernel_volume;
  int _kernel_idx = tidx % kernel_volume;
  int kernel_idx = _kernel_idx;
  if (idx >= n_points) return;

  int coords_out[NDim];
  coords_out[0] = in_coords[idx * NDim];  //batch_idx
  
  #pragma unroll
  for(int i = NDim - 1; i > 0; i--){
    int cur_offset = _kernel_idx % kernel_sizes[i - 1];
    // cur_offset -= (kernel_sizes[i - 1] - 1);  //shift the kernel offset to <= 0
    coords_out[i] = in_coords[idx * NDim + i] + cur_offset;
    _kernel_idx /= kernel_sizes[i - 1];
  }
  
  if (coords_out[1] >= coords_min[1] &&
    coords_out[1] <= coords_max[1] &&
    coords_out[2] >= coords_min[2] &&
    coords_out[2] <= coords_max[2] &&
    coords_out[3] >= coords_min[3] &&
    coords_out[3] <= coords_max[3]) {
  
    type_int grid_index = transform_coords<type_int>(coords_out, coords_min, coords_max);
    int input_idx = table.lookup(grid_index + 1) - 1;
    if (input_idx >= 0) {
      out_in_map[idx * kernel_volume + kernel_idx] = input_idx;
      // out_in_map[input_idx * kernel_volume + kernel_volume - 1 - kernel_idx] = idx;
    }
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


std::vector<at::Tensor> build_kernel_map_subm_hashmap_int32(
    hashtable32& table,
    at::Tensor _in_coords, at::Tensor _coords_min, at::Tensor _coords_max,
    at::Tensor _kernel_sizes, at::Tensor _stride,
    at::Tensor _padding, bool to_insert) {
  int n_points = _in_coords.size(0);
  int kernel_volume = (int)(torch::prod(_kernel_sizes).item<int>());
  int *in_coords = _in_coords.data_ptr<int>();
  int *coords_min = _coords_min.data_ptr<int>();
  int *coords_max = _coords_max.data_ptr<int>();
  int *kernel_sizes = _kernel_sizes.data_ptr<int>();
  int *stride = _stride.data_ptr<int>();
  auto options = torch::TensorOptions()
                     .dtype(at::ScalarType::Int)
                     .device(_in_coords.device());
  // auto options_long =
  // torch::TensorOptions().dtype(at::ScalarType::Long).device(_in_coords.device());
  at::Tensor _out_coords = torch::empty({_in_coords.size(0)}, options);
  int32_t *out_coords = _out_coords.data_ptr<int>();
  int divisor = table.get_divisor();
  int n_points_pad = (n_points + divisor - 1) / divisor * divisor;
  at::Tensor _out_in_map = torch::full({n_points_pad, kernel_volume}, -1, options);
  int *out_in_map = _out_in_map.data_ptr<int>();
  // stage1: insert to hashmap
  if (to_insert)
    subm_hashmap_kmap_stage1<hashtable32::device_view, int32_t><<<(int)ceil((double)n_points / 256), 256>>>(
        table.get_device_view(), n_points, kernel_volume, in_coords, coords_min, coords_max, out_coords);
  // stage2: query
  if (kernel_volume % 2 != 0){
    subm_hashmap_kmap_stage2_odd_kernel<hashtable32::device_view, int32_t><<<(int)ceil((double)n_points * (kernel_volume / 2) / 256), 256>>>(
        table.get_device_view(), n_points, kernel_volume, in_coords, coords_min, coords_max,
        kernel_sizes, out_in_map);  // only support odd kernel shapes
  }
  else {
    subm_hashmap_kmap_stage2_even_kernel<hashtable32::device_view, int32_t><<<(int)ceil((double)n_points * (kernel_volume) / 256), 256>>>(
        table.get_device_view(), n_points, kernel_volume, in_coords, coords_min, coords_max,
        kernel_sizes, out_in_map);  // only support even kernel shapes
  }


  return {_out_in_map};
}


std::vector<at::Tensor> build_kernel_map_subm_hashmap(
    hashtable& table,
    at::Tensor _in_coords, at::Tensor _coords_min, at::Tensor _coords_max,
    at::Tensor _kernel_sizes, at::Tensor _stride,
    at::Tensor _padding, bool to_insert) {
  int n_points = _in_coords.size(0);
  int kernel_volume = (int)(torch::prod(_kernel_sizes).item<int>());
  int *in_coords = _in_coords.data_ptr<int>();
  int *coords_min = _coords_min.data_ptr<int>();
  int *coords_max = _coords_max.data_ptr<int>();
  int *kernel_sizes = _kernel_sizes.data_ptr<int>();
  int *stride = _stride.data_ptr<int>();
  auto options = torch::TensorOptions()
                     .dtype(at::ScalarType::Int)
                     .device(_in_coords.device());
  auto options_long = torch::TensorOptions()
                     .dtype(at::ScalarType::Long)
                     .device(_in_coords.device());
  // auto options_long =
  // torch::TensorOptions().dtype(at::ScalarType::Long).device(_in_coords.device());
  at::Tensor _out_coords = torch::empty({_in_coords.size(0)}, options_long);
  int64_t *out_coords = _out_coords.data_ptr<int64_t>();
  int divisor = table.get_divisor();
  int n_points_pad = (n_points + divisor - 1) / divisor * divisor;
  at::Tensor _out_in_map = torch::full({n_points_pad, kernel_volume}, -1, options);
  int *out_in_map = _out_in_map.data_ptr<int>();
  // stage1: insert to hashmap
  if (to_insert)
    subm_hashmap_kmap_stage1<hashtable::device_view, int64_t><<<(int)ceil((double)n_points / 256), 256>>>(
        table.get_device_view(), n_points, kernel_volume, in_coords, coords_min, coords_max, out_coords);
  // stage2: query
  if (kernel_volume % 2 != 0){
    subm_hashmap_kmap_stage2_odd_kernel<hashtable::device_view, int64_t><<<(int)ceil((double)n_points * (kernel_volume / 2) / 256), 256>>>(
        table.get_device_view(), n_points, kernel_volume, in_coords, coords_min, coords_max,
        kernel_sizes, out_in_map);  // only support odd kernel shapes
  }
  else {
    subm_hashmap_kmap_stage2_even_kernel<hashtable::device_view, int64_t><<<(int)ceil((double)n_points * (kernel_volume) / 256), 256>>>(
        table.get_device_view(), n_points, kernel_volume, in_coords, coords_min, coords_max,
        kernel_sizes, out_in_map);  // only support even kernel shapes
  }
  return {_out_in_map};
}

std::vector<at::Tensor> build_kernel_map_downsample_hashmap_int32(
    hashtable32& table,
    at::Tensor _in_coords, at::Tensor _coords_min, at::Tensor _coords_max,
    at::Tensor _kernel_sizes, at::Tensor _stride,
    at::Tensor _padding, bool to_insert) {
  int n_points = _in_coords.size(0);
  int kernel_volume = (int)(torch::prod(_kernel_sizes).item<int>());
  int *in_coords = _in_coords.data_ptr<int>();
  int *coords_min = _coords_min.data_ptr<int>();
  int *coords_max = _coords_max.data_ptr<int>();
  int *kernel_sizes = _kernel_sizes.data_ptr<int>();
  int *stride = _stride.data_ptr<int>();
  int *padding = _padding.data_ptr<int>();
  auto options = torch::TensorOptions()
                     .dtype(at::ScalarType::Int)
                     .device(_in_coords.device());
  auto options_long = torch::TensorOptions()
                          .dtype(at::ScalarType::Int)
                          .device(_in_coords.device());

  at::Tensor _out_kmap = torch::full({n_points, kernel_volume}, -1, options);

  at::Tensor _n_out_points = torch::zeros({1}, options);
  at::Tensor _transformed_out_coords =
      torch::empty({kernel_volume * n_points}, options);
  // transformed coordinates is long
  int32_t *out_kmap = _out_kmap.data_ptr<int>();
  int *n_out_points = _n_out_points.data_ptr<int>();
  int32_t *transformed_out_coords = _transformed_out_coords.data_ptr<int>();
  /*
  // If we do specialized downsample for 3D coords (stage 1), we do it (using
  divided coords_min/max) as follows:
  */
  if (kernel_volume % 2 == 1)
  {
    downsample_grid_kmap_stage1_specialized_fast<int32_t, true><<<(int)ceil((double)(n_points * kernel_volume) / 256),
                                              256>>>(
        n_points, kernel_volume, in_coords, kernel_sizes, stride,
        padding, coords_min, coords_max, n_out_points, transformed_out_coords, out_kmap);
  }
  else
  {
    downsample_grid_kmap_stage1_specialized_fast<int32_t, false><<<(int)ceil((double)(n_points * kernel_volume) / 256),
                                              256>>>(
        n_points, kernel_volume, in_coords, kernel_sizes, stride,
        padding, coords_min, coords_max, n_out_points, transformed_out_coords, out_kmap);
  }
  // stage2: get unique coordinates and insert them to the grid.
  int n_out_points_with_duplicate = _n_out_points.item<int>();
  at::Tensor _out_coords = std::get<0>(torch::_unique(torch::from_blob(transformed_out_coords, {n_out_points_with_duplicate}, options)));
  int32_t *out_coords = _out_coords.data_ptr<int>();
  // stage 2.1: insert to the hashmap and transform the out coords to N x 4 format.
  int n_out_points_scalar = _out_coords.size(0);
  // Check the _capacity of hashtable
  int capacity = table.get_capacity();
  if (capacity < n_out_points_scalar)
    throw std::invalid_argument("The capacity of hashtable is not sufficient. Please enlarge reserved space for hashtable:\n # Python \nimport torchsparse.backends\ntorchsparse.backends.hash_rsv_ratio=#Value");

  at::Tensor final_out_coords =
      torch::zeros({n_out_points_scalar, NDim}, options);
  inverse_transform_coords_and_insert_kernel<<<
      (int)ceil((double)n_out_points_scalar / 256), 256>>>(
      table.get_device_view(), n_out_points_scalar, out_coords, 
      coords_min, coords_max, final_out_coords.data_ptr<int>());
  
  //table.insert_vals(_out_coords);

  // stage3: replace the (64b) coordinate ravel hashes with the output idx  
  int divisor = table.get_divisor();
  at::Tensor _out_in_map =
      torch::full({(n_out_points_scalar + divisor - 1) / divisor * divisor, kernel_volume}, -1, options);
  int *out_in_map = _out_in_map.data_ptr<int>();
  
  downsample_hashmap_kmap_stage3<<<
      (int)ceil((double)(n_points * kernel_volume) / 256), 256>>>(
      table.get_device_view(), n_points, n_out_points_scalar, kernel_volume, out_kmap,
      out_in_map);
  
  return {_out_in_map, final_out_coords};
}


std::vector<at::Tensor> build_kernel_map_downsample_hashmap(
    hashtable& table,
    at::Tensor _in_coords, at::Tensor _coords_min, at::Tensor _coords_max,
    at::Tensor _kernel_sizes, at::Tensor _stride,
    at::Tensor _padding, bool to_insert) {
  int n_points = _in_coords.size(0);
  int kernel_volume = (int)(torch::prod(_kernel_sizes).item<int>());
  int *in_coords = _in_coords.data_ptr<int>();
  int *coords_min = _coords_min.data_ptr<int>();
  int *coords_max = _coords_max.data_ptr<int>();
  int *kernel_sizes = _kernel_sizes.data_ptr<int>();
  int *stride = _stride.data_ptr<int>();
  int *padding = _padding.data_ptr<int>();
  auto options = torch::TensorOptions()
                     .dtype(at::ScalarType::Int)
                     .device(_in_coords.device());
  auto options_long = torch::TensorOptions()
                          .dtype(at::ScalarType::Long)
                          .device(_in_coords.device());

  at::Tensor _out_kmap = torch::full({n_points, kernel_volume}, -1, options_long);
  at::Tensor _n_out_points = torch::zeros({1}, options);
  at::Tensor _transformed_out_coords =
      torch::empty({kernel_volume * n_points}, options_long);
  // transformed coordinates is long
  int64_t *out_kmap = _out_kmap.data_ptr<int64_t>();
  int *n_out_points = _n_out_points.data_ptr<int>();
  int64_t *transformed_out_coords = _transformed_out_coords.data_ptr<int64_t>();
  /*
  // If we do specialized downsample for 3D coords (stage 1), we do it (using
  divided coords_min/max) as follows:
  */
   
  if (kernel_volume % 2 == 1)
  { 
    downsample_grid_kmap_stage1_specialized_fast<int64_t, true><<<(int)ceil((double)(n_points * kernel_volume) / 256),
                                              256>>>(
        n_points, kernel_volume, in_coords, kernel_sizes, stride,
        padding, coords_min, coords_max, n_out_points, transformed_out_coords, out_kmap);
  }
  else
  {
    downsample_grid_kmap_stage1_specialized_fast<int64_t, false><<<(int)ceil((double)(n_points * kernel_volume) / 256),
                                              256>>>(
        n_points, kernel_volume, in_coords, kernel_sizes, stride,
        padding, coords_min, coords_max, n_out_points, transformed_out_coords, out_kmap);
  }
  // stage2: get unique coordinates and insert them to the grid.
  int n_out_points_with_duplicate = _n_out_points.item<int>();
  at::Tensor _out_coords = std::get<0>(torch::_unique(torch::from_blob(transformed_out_coords, {n_out_points_with_duplicate}, options_long)));
  int64_t *out_coords = _out_coords.data_ptr<int64_t>();

  // stage 2.1: insert to the hashmap and transform the out coords to N x 4 format.
  int n_out_points_scalar = _out_coords.size(0);
  // Check the _capacity of hashtable
  int capacity = table.get_capacity();
  if (capacity < n_out_points_scalar)
    throw std::invalid_argument("The capacity of hashtable is not sufficient. Please enlarge reserved space for hashtable:\n # Python \nimport torchsparse.backends\ntorchsparse.backends.hash_rsv_ratio=#Value");

  at::Tensor final_out_coords =
      torch::zeros({n_out_points_scalar, NDim}, options);
  inverse_transform_coords_and_insert_kernel<<<
      (int)ceil((double)n_out_points_scalar / 256), 256>>>(
      table.get_device_view(), n_out_points_scalar, out_coords, 
      coords_min, coords_max, final_out_coords.data_ptr<int>());
  //table.insert_vals(_out_coords);

  // stage3: replace the (64b) coordinate ravel hashes with the output idx  
  int divisor = table.get_divisor();
  at::Tensor _out_in_map =
      torch::full({(n_out_points_scalar + divisor - 1) / divisor * divisor, kernel_volume}, -1, options);
  int *out_in_map = _out_in_map.data_ptr<int>();
  
  downsample_hashmap_kmap_stage3<<<
      (int)ceil((double)(n_points * kernel_volume) / 256), 256>>>(
      table.get_device_view(), n_points, n_out_points_scalar, kernel_volume, out_kmap,
      out_in_map);
  return {_out_in_map, final_out_coords};
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

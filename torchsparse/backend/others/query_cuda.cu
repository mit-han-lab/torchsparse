#include <torch/torch.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "../hashmap/hashmap_cuda.cuh"

__global__ void convert_out_in_map_kernel(const int* out_in_map, int* out_in_map_t, int n, int kernel_volume){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= n * kernel_volume) return;
  int input_idx = out_in_map[idx];
  if(input_idx < 0) return;
  out_in_map_t[idx % kernel_volume + input_idx * kernel_volume] = idx / kernel_volume;
}

__global__ void derive_bit_mask_from_out_in_map_kernel(int* out_in_map, int* bitmask, int valid_n, int n, int kernel_volume, int split_mask_num){
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = tidx / split_mask_num;
  if(idx >= valid_n) return;
  int split_mask_iter = tidx % split_mask_num;
  int split_mask_len = (kernel_volume + split_mask_num - 1) / split_mask_num;
  int* cur_out_in_map = out_in_map + kernel_volume * idx + split_mask_iter * split_mask_len;
  if (split_mask_iter == (split_mask_num - 1)) // The last tile
    split_mask_len = kernel_volume - split_mask_iter * split_mask_len;
  int cur_bitmask = 0;
  for(int i = 0; i < split_mask_len; i++){
    cur_bitmask += (int)(cur_out_in_map[i] >= 0) * (int)(1u << i);
  }
  bitmask[split_mask_iter * n + idx] = cur_bitmask;
}

at::Tensor hash_query_cuda(const at::Tensor hash_query,
                           const at::Tensor hash_target,
                           const at::Tensor idx_target) {
  // return group_point_forward_gpu(points, indices);
  int n = hash_target.size(0);
  int n1 = hash_query.size(0);
  hashtable in_hash_table(n * 2);

  in_hash_table.insert_many(hash_target.data_ptr<int64_t>(), n);

  at::Tensor out = torch::zeros(
      {n1}, at::device(hash_query.device()).dtype(at::ScalarType::Int));
  in_hash_table.lookup_many(hash_query.data_ptr<int64_t>(), out.data_ptr<int>(), n1);
  return out;
}


void convert_transposed_out_in_map(const at::Tensor out_in_map,
                            at::Tensor out_in_map_t) {
  convert_out_in_map_kernel<<<(out_in_map.size(0) * out_in_map.size(1) + 255) / 256, 256>>>(
    out_in_map.data_ptr<int>(), out_in_map_t.data_ptr<int>(), out_in_map.size(0), out_in_map.size(1));
}




at::Tensor derive_bitmask_from_out_in_map(const at::Tensor out_in_map, const int split_mask_num, int valid_n) {
  at::Tensor bitmask = torch::full(
      {split_mask_num, out_in_map.size(0)}, -1, at::device(out_in_map.device()).dtype(at::ScalarType::Int));
  derive_bit_mask_from_out_in_map_kernel<<<(split_mask_num * out_in_map.size(0) + 255) / 256, 256>>>(
    out_in_map.data_ptr<int>(), bitmask.data_ptr<int>(), valid_n, out_in_map.size(0), out_in_map.size(1), split_mask_num);
  return bitmask;
}

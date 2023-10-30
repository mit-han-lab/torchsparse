#include <torch/extension.h>
#include "reorder_map_cuda.h"

#define cta_M 128
#define thd_num 128 // 1 thd per row


__global__ void __launch_bounds__(thd_num) reorder_out_in_map_kernel(
    int* __restrict__ out_in_map,
    int* __restrict__ reorder_loc, 
    int M, // node num
    int kernel_volume,
    int split_mask_len,
    int* __restrict__ reorder_out_in_map
){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int output_row_idx = index / kernel_volume;
  int output_col_idx = index % kernel_volume;
  if (output_row_idx >= M) return;
  int split_mask_iter = output_col_idx / split_mask_len;
  int input_row_idx = reorder_loc[split_mask_iter * M + output_row_idx];
  reorder_out_in_map[output_row_idx * kernel_volume + output_col_idx] = out_in_map[input_row_idx * kernel_volume + output_col_idx];
}

at::Tensor reorder_out_in_map_cuda(
    torch::Tensor _out_in_map,
    torch::Tensor _reorder_loc
){

    int M = _out_in_map.size(0);
    int kernel_volume = _out_in_map.size(1);
    int split_mask_num = _reorder_loc.size(0);
    int split_mask_len = (kernel_volume + split_mask_num - 1) / split_mask_num;

    auto options =
      torch::TensorOptions().dtype(_out_in_map.dtype()).device(_out_in_map.device());
    at::Tensor _reorder_out_in_map = torch::empty({M, kernel_volume}, options);


    auto out_in_map = _out_in_map.data_ptr<int>();
    auto reorder_loc = _reorder_loc.data_ptr<int>();
    auto reorder_out_in_map = _reorder_out_in_map.data_ptr<int>();

    reorder_out_in_map_kernel<<<(M + cta_M - 1) / cta_M * kernel_volume, cta_M>>>(
        out_in_map, reorder_loc, M, kernel_volume, split_mask_len, reorder_out_in_map);
    
    return _reorder_out_in_map;
} 
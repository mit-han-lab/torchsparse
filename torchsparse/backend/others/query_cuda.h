#pragma once

#include <torch/torch.h>

at::Tensor hash_query_cuda(const at::Tensor hash_query,
                           const at::Tensor hash_target,
                           const at::Tensor idx_target);
void convert_transposed_out_in_map(const at::Tensor out_in_map,
                            at::Tensor out_in_map_t);
at::Tensor derive_bitmask_from_out_in_map(const at::Tensor out_in_map, const int split_mask_num, int valid_n);
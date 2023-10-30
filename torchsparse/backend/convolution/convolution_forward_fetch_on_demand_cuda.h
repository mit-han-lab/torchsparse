#pragma once

#include <torch/torch.h>

at::Tensor conv_forward_fetch_on_demand_cuda(
    at::Tensor in_feat, at::Tensor kernel, 
    at::Tensor neighbor_map, const int sum_nnz, 
    at::Tensor neighbor_address, at::Tensor q_neighbor_address,
    const int output_size, const int qsum_nnz, const bool transpose, 
    const bool allow_tf32, const bool allow_fp16);

at::Tensor conv_forward_fetch_on_demand_no_fusion_cuda(
    at::Tensor in_feat, at::Tensor kernel,
    at::Tensor neighbor_map, at::Tensor neighbor_offset, 
    const int sum_nnz, const int output_size, const bool transpose, 
    const bool allow_tf32, const bool allow_fp16);
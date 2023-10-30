#pragma once

#include <torch/torch.h>

at::Tensor conv_forward_gather_scatter_cuda(
    at::Tensor in_feat, at::Tensor kernel, at::Tensor neighbor_map,
    at::Tensor neighbor_offset, at::Tensor input_mask, at::Tensor output_mask,
    const int output_size, const float epsilon, const int mm_thresh,
    const int conv_mode, const bool transpose, at::Tensor buffer);

at::Tensor conv_forward_gather_scatter_cuda_latest(
    at::Tensor in_feat, at::Tensor kernel, at::Tensor neighbor_map,
    at::Tensor neighbor_offset, at::Tensor input_mask, at::Tensor output_mask,
    const int output_size, const float epsilon, const int mm_thresh,
    const int conv_mode, const bool transpose, at::Tensor buffer);

at::Tensor conv_forward_gather_scatter_cuda_fallback(
    at::Tensor in_feat, at::Tensor kernel, at::Tensor neighbor_map,
    const int output_size, const int conv_mode, at::Tensor neighbor_offset,
    const bool transpose);

void conv_backward_gather_scatter_cuda(at::Tensor in_feat, at::Tensor grad_in_feat,
                               at::Tensor grad_out_feat, at::Tensor kernel,
                               at::Tensor grad_kernel, at::Tensor neighbor_map,
                               at::Tensor neighbor_offset,
                               const bool transpose);

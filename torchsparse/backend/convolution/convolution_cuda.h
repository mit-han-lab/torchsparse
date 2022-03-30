#pragma once

#include <torch/torch.h>

void convolution_forward_cuda(at::Tensor in_feat, at::Tensor out_feat,
                              at::Tensor kernel, at::Tensor neighbor_map,
                              at::Tensor neighbor_offset, at::Tensor input_mask,
                              at::Tensor output_mask, const float epsilon,
                              const int mm_thresh, const int conv_mode,
                              const bool transpose, at::Tensor buffer);

void convolution_forward_cuda_latest(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor kernel,
    at::Tensor neighbor_map, at::Tensor neighbor_offset, at::Tensor input_mask,
    at::Tensor output_mask, const float epsilon, const int mm_thresh,
    const int conv_mode, const bool transpose, at::Tensor buffer);

void convolution_forward_cuda_fallback(at::Tensor in_feat, at::Tensor out_feat,
                                       at::Tensor kernel,
                                       at::Tensor neighbor_map,
                                       const int conv_mode,
                                       at::Tensor neighbor_offset,
                                       const bool transpose);

void convolution_backward_cuda(at::Tensor in_feat, at::Tensor grad_in_feat,
                               at::Tensor grad_out_feat, at::Tensor kernel,
                               at::Tensor grad_kernel, at::Tensor neighbor_map,
                               at::Tensor neighbor_offset,
                               const bool transpose);

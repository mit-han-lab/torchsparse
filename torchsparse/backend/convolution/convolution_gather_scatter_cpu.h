#pragma once

#include <torch/torch.h>

void conv_forward_gather_scatter_cpu(at::Tensor in_feat, at::Tensor out_feat,
                             at::Tensor kernel, at::Tensor neighbor_map,
                             at::Tensor neighbor_offset, const bool transpose);

void conv_backward_gather_scatter_cpu(at::Tensor in_feat, at::Tensor grad_in_feat,
                              at::Tensor grad_out_feat, at::Tensor kernel,
                              at::Tensor grad_kernel, at::Tensor neighbor_map,
                              at::Tensor neighbor_offset, const bool transpose);

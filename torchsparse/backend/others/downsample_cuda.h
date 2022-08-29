#pragma once

#include <torch/torch.h>

at::Tensor downsample_cuda(at::Tensor in_coords, at::Tensor coords_max,
                           at::Tensor coords_min, std::vector<int> kernel_sizes,
                           std::vector<int> stride,
                           std::vector<int> tensor_stride);

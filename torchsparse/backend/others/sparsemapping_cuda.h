#pragma once

#include <torch/torch.h>

std::vector<at::Tensor> build_kernel_map_subm(
    at::Tensor in_coords, at::Tensor coords_min, at::Tensor coords_max,
    at::Tensor kernel_sizes, at::Tensor stride, at::Tensor tensor_stride);

std::vector<at::Tensor> build_kernel_map_downsample(
    at::Tensor in_coords, at::Tensor coords_min, at::Tensor coords_max,
    at::Tensor kernel_sizes, at::Tensor stride, at::Tensor tensor_stride);

std::vector<at::Tensor> build_mask_from_kmap(int n_points, int n_out_points,
                                             at::Tensor _kmap,
                                             at::Tensor _kmap_sizes);

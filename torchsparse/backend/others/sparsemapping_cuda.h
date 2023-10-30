#pragma once

#include <torch/torch.h>
#include "../hashmap/hashmap_cuda.cuh"

std::vector<at::Tensor> build_mask_from_kmap(int n_points, int n_out_points,
                                             at::Tensor _kmap,
                                             at::Tensor _kmap_sizes);

std::vector<at::Tensor> build_kernel_map_subm_hashmap(
    hashtable& table,
    at::Tensor _in_coords, at::Tensor _coords_min, at::Tensor _coords_max,
    at::Tensor _kernel_sizes, at::Tensor _stride,
    at::Tensor padding, bool to_insert);

std::vector<at::Tensor> build_kernel_map_downsample_hashmap(
    hashtable& table,
    at::Tensor _in_coords, at::Tensor _coords_min, at::Tensor _coords_max,
    at::Tensor _kernel_sizes, at::Tensor _stride,
    at::Tensor _padding, bool to_insert);

std::vector<at::Tensor> build_kernel_map_subm_hashmap_int32(
    hashtable32& table,
    at::Tensor _in_coords, at::Tensor _coords_min, at::Tensor _coords_max,
    at::Tensor _kernel_sizes, at::Tensor _stride,
    at::Tensor padding, bool to_insert);

std::vector<at::Tensor> build_kernel_map_downsample_hashmap_int32(
    hashtable32& table,
    at::Tensor _in_coords, at::Tensor _coords_min, at::Tensor _coords_max,
    at::Tensor _kernel_sizes, at::Tensor _stride,
    at::Tensor _padding, bool to_insert);

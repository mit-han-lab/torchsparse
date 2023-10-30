/*
Please consider citing the following paper when using the code:

@inproceedings{hong2023pcengine,
  title={{Exploiting Hardware Utilization and Adaptive Dataflow for Efficient Sparse Convolution in 3D Point Clouds}},
  author={Hong, Ke and Yu, Zhongming and Dai, Guohao and Yang, Xinhao and Lian, Yaoxiu and Liu, Zehao and Xu, Ningyi and Wang, Yu},
  booktitle={Sixth Conference on Machine Learning and Systems (MLSys)},
  year={2023}
}
*/

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

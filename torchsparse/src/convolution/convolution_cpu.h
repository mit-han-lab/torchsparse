#ifndef _SPARSE_CONVOLUTION_CPU
#define _SPARSE_CONVOLUTION_CPU
#include <torch/extension.h>
#include <chrono>
#include <algorithm>

void cpu_scatter_launch(const int n_in, const int n_out, const int c, 
                               const float *in_feat, float *out_feat, 
                               const int *kmap, const bool transpose);

void cpu_gather_launch(const int n_k, const int n_in, const int c, 
                               const float *in_feat, float *out_feat, 
                               const int *kmap, const bool transpose);

void ConvolutionForwardCPU(at::Tensor in_feat, at::Tensor out_feat,
                           at::Tensor kernel, at::Tensor neighbor_map, 
                           at::Tensor neighbor_offset, const bool transpose);

void ConvolutionBackwardCPU(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor kernel, at::Tensor grad_kernel, at::Tensor neighbor_map,
    at::Tensor neighbor_offset, const bool transpose);

#endif
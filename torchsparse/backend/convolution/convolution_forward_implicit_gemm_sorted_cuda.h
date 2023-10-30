#include <torch/extension.h>

at::Tensor conv_forward_implicit_gemm_sorted_cuda(
                       torch::Tensor _in_feats, torch::Tensor _kernel,
                       torch::Tensor _out_in_map,torch::Tensor _reduced_mask,
                       torch::Tensor _reorder_loc, 
                       int num_out_feats, int num_out_channels,
                       bool allow_tf32, bool allow_fp16);

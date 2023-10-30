#include <torch/extension.h>

at::Tensor conv_backward_wgrad_implicit_gemm_cuda(
                       torch::Tensor _in_feats, torch::Tensor _kernel,
                       torch::Tensor _out_in_map, const int split_k_iters,
                       bool allow_tf32, bool allow_fp16);

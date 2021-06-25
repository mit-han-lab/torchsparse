#ifndef TORCHSPARSE_VOXELIZE_CUDA
#define TORCHSPARSE_VOXELIZE_CUDA

#include <torch/torch.h>

at::Tensor voxelize_forward_cuda(const at::Tensor inputs, const at::Tensor idx,
                                 const at::Tensor counts);

at::Tensor voxelize_backward_cuda(const at::Tensor top_grad,
                                  const at::Tensor idx, const at::Tensor counts,
                                  const int N);

#endif

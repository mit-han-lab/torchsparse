#ifndef TORCHSPARSE_HASH_CUDA
#define TORCHSPARSE_HASH_CUDA

#include <torch/torch.h>

at::Tensor hash_cuda(const at::Tensor idx);

at::Tensor kernel_hash_cuda(const at::Tensor idx,
                            const at::Tensor kernel_offset);

#endif

#ifndef _SPARSE_HASH_CPU
#define _SPARSE_HASH_CPU

#include <torch/torch.h>

at::Tensor hash_cpu(const at::Tensor idx);

at::Tensor kernel_hash_cpu(const at::Tensor idx,
                           const at::Tensor kernel_offset);

#endif

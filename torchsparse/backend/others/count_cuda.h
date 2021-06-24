#ifndef _SPARSE_COUNT
#define _SPARSE_COUNT

#include <torch/torch.h>

at::Tensor count_cuda(const at::Tensor idx, const int s);

#endif

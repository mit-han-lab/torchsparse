#ifndef _SPARSE_QUERY
#define _SPARSE_QUERY

#include <torch/torch.h>

at::Tensor hash_query_cuda(const at::Tensor hash_query,
                           const at::Tensor hash_target,
                           const at::Tensor idx_target);

#endif

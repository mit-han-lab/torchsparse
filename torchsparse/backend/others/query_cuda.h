#pragma once

#include <torch/torch.h>

at::Tensor hash_query_cuda(const at::Tensor hash_query,
                           const at::Tensor hash_target,
                           const at::Tensor idx_target);

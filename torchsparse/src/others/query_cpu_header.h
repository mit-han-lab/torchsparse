#ifndef _SPARSE_QUERY_CPU
#define _SPARSE_QUERY_CPU
#include <torch/torch.h>
#include <vector>
#include <cmath>
#include <iostream>

at::Tensor cpu_query_forward(
    const at::Tensor hash_query,
    const at::Tensor hash_target,
    const at::Tensor idx_target
);
#endif
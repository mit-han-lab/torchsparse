#ifndef _SPARSE_QUERY
#define _SPARSE_QUERY
#include <torch/torch.h>
#include <vector>
#include <cmath>
#include <iostream>

std::vector<at::Tensor> query_forward(
    const at::Tensor hash_query,
    const at::Tensor hash_target,
    const at::Tensor idx_target
);
#endif
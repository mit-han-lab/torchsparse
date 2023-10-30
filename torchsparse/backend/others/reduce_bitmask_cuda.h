#pragma once
#include <torch/torch.h>

torch::Tensor reduce_bitmask_cuda(
    torch::Tensor _bitmask_int,
    int M_tile
);
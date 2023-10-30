#pragma once
#include <torch/torch.h>

at::Tensor reorder_out_in_map_cuda(
    torch::Tensor _out_in_map,
    torch::Tensor _reorder_loc
);
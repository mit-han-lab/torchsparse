#pragma once

#include <torch/torch.h>

at::Tensor exclusive_scan_quantified_wrapper(
    const int k_vol, at::Tensor neighbor_offset, 
    at::Tensor neighbor_address, at::Tensor q_neighbor_address);
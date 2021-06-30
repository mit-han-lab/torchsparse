#pragma once

#include <torch/torch.h>

at::Tensor count_cuda(const at::Tensor idx, const int s);

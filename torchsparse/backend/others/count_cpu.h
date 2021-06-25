#pragma once

#include <torch/torch.h>

at::Tensor count_cpu(const at::Tensor idx, const int s);

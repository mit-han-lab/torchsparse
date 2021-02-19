#ifndef _COUNT_CPU
#define _COUNT_CPU
#include <torch/torch.h>

at::Tensor count_forward_cpu(
    const at::Tensor idx,
    const int s);

#endif
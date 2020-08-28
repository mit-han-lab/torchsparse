#ifndef _SPARSE_DEVOXELIZE_CPU
#define _SPARSE_DEVOXELIZE_CPU
#include <torch/torch.h>
#include <vector>

at::Tensor cpu_devoxelize_forward(
    const at::Tensor feat,
    const at::Tensor indices,
    const at::Tensor weight
);
at::Tensor cpu_devoxelize_backward(
    const at::Tensor top_grad,
    const at::Tensor indices,
    const at::Tensor weight,
    int n
);

#endif
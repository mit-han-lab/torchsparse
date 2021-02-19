#ifndef _SPARSE_INSERT_CPU
#define _SPARSE_INSERT_CPU
#include <torch/torch.h>
#include <vector>


at::Tensor cpu_insertion_forward(
    const at::Tensor inputs,
    const at::Tensor idx,
    const at::Tensor counts
);
at::Tensor cpu_insertion_backward(
    const at::Tensor top_grad,
    const at::Tensor idx,
    const at::Tensor counts,
    const int N
);
#endif
#ifndef _SPARSE_HASH_CPU
#define _SPARSE_HASH_CPU
#include <torch/torch.h>
#include <vector>

void cpu_hash_wrapper(int N, const int *data, long *out);
void cpu_kernel_hash_wrapper(int N, int K, const int *data, const int *offsets, long int *out);
at::Tensor cpu_hash_forward(
    const at::Tensor idx);
at::Tensor cpu_kernel_hash_forward(
    const at::Tensor idx,
    const at::Tensor offsets);
#endif
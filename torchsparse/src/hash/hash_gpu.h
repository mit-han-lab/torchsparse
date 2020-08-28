#ifndef _SPARSE_HASH
#define _SPARSE_HASH
#include <torch/torch.h>
#include <vector>

//CUDA forward declarations
void hash_wrapper(int N, const int * data, long * out);
void kernel_hash_wrapper(int N, int K, const int * data, const int *kernel_offset, long int * out);
at::Tensor hash_forward(
    const at::Tensor idx
);
at::Tensor kernel_hash_forward(
    const at::Tensor idx,
    const at::Tensor kernel_offset
);
#endif
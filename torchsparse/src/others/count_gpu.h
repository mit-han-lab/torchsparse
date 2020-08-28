#ifndef _SPARSE_COUNT
#define _SPARSE_COUNT
#include <torch/torch.h>
#include <vector>

//CUDA forward declarations
void count_wrapper(int N, const int * data, int * out);
at::Tensor count_forward(
    const at::Tensor idx,
    const int s
);
#endif
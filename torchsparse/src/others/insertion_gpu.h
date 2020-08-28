#ifndef _SPARSE_INSERT
#define _SPARSE_INSERT
#include <torch/torch.h>
#include <vector>


//CUDA forward declarations
void insertion_wrapper(int N, int c, int s, const float * data, const int * idx, const int * counts, float * out);
void insertion_grad_wrapper(int N, int c, int s, const float * top_grad, const int * idx, const int * counts, float * bottom_grad);
//make sure indices is int type
//feat: (b,c,n) indices: (b,n) -> out: (b,c,s), out_indices: (b,n) (preprocessed indices)

at::Tensor insertion_forward(
    const at::Tensor inputs,
    const at::Tensor idx,
    const at::Tensor counts
);
at::Tensor insertion_backward(
    const at::Tensor top_grad,
    const at::Tensor idx,
    const at::Tensor counts,
    const int N
);
#endif
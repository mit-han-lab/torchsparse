#ifndef _SPARSE_INSERT
#define _SPARSE_INSERT
#include <torch/torch.h>
#include <vector>

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
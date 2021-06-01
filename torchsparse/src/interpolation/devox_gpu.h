#ifndef _SPARSE_DEVOXELIZE
#define _SPARSE_DEVOXELIZE
#include <torch/torch.h>
#include <vector>

//CUDA forward declarations
void deterministic_devoxelize_wrapper(int N, int c, const int *indices, const float *weight, const float *feat, float *out);
void deterministic_devoxelize_grad_wrapper(int N, int n, int c, const int *indices, const float *weight, const float *top_grad, int *bottom_grad);
at::Tensor devoxelize_forward(
    const at::Tensor feat,
    const at::Tensor indices,
    const at::Tensor weight);
at::Tensor devoxelize_backward(
    const at::Tensor top_grad,
    const at::Tensor indices,
    const at::Tensor weight,
    int n);
at::Tensor deterministic_devoxelize_forward(
    const at::Tensor feat,
    const at::Tensor indices,
    const at::Tensor weight);
at::Tensor deterministic_devoxelize_backward(
    const at::Tensor top_grad,
    const at::Tensor indices,
    const at::Tensor weight,
    int n);
#endif
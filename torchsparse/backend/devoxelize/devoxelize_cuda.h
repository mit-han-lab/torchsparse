#pragma once

#include <torch/torch.h>

at::Tensor devoxelize_forward_cuda(const at::Tensor feat,
                                   const at::Tensor indices,
                                   const at::Tensor weight);

at::Tensor devoxelize_backward_cuda(const at::Tensor top_grad,
                                    const at::Tensor indices,
                                    const at::Tensor weight, int n);

#include <torch/torch.h>
#include <vector>
#include "devox_gpu.h"

//make sure indices is int type
//feat: (b,c,s) indices: (N, 3) batch_index: (N, ) -> out: (N, c)
at::Tensor deterministic_devoxelize_forward(
    const at::Tensor feat,
    const at::Tensor indices,
    const at::Tensor weight)
{
  int b = feat.size(0);
  //printf("%d\n", b);
  int c = feat.size(1);
  int N = indices.size(0);

  at::Tensor out = torch::zeros({N, c}, at::device(feat.device()).dtype(at::ScalarType::Float));
  deterministic_devoxelize_wrapper(N, c, indices.data_ptr<int>(), weight.data_ptr<float>(), feat.data_ptr<float>(), out.data_ptr<float>());
  return out;
}

//top_grad: (N, c), indices: (N, 3), batch_index: (N, ) -> bottom_grad: (b,c,s), s=r^3
at::Tensor deterministic_devoxelize_backward(
    const at::Tensor top_grad,
    const at::Tensor indices,
    const at::Tensor weight,
    int n)
{
  int c = top_grad.size(1);
  int N = top_grad.size(0);
  at::Tensor bottom_grad_int = torch::zeros({n, c}, at::device(top_grad.device()).dtype(at::ScalarType::Int));
  deterministic_devoxelize_grad_wrapper(N, n, c, indices.data_ptr<int>(), weight.data_ptr<float>(), top_grad.data_ptr<float>(), bottom_grad_int.data_ptr<int>());

  at::Tensor bottom_grad = bottom_grad_int.to(at::ScalarType::Double);
  //std::cout << torch::mean(bottom_grad) << std::endl;
  bottom_grad /= 1e10;
  return bottom_grad.to(at::ScalarType::Float);
}

#include <torch/torch.h>
#include <vector>
#include "insertion_cpu_header.h"

at::Tensor cpu_insertion_forward(
    const at::Tensor inputs,
    const at::Tensor idx,
    const at::Tensor counts)
{
  //return group_point_forward_gpu(points, indices);

  int N = inputs.size(0);
  int c = inputs.size(1);
  int N1 = counts.size(0);
  at::Tensor out = torch::zeros({N1, c}, at::device(idx.device()).dtype(at::ScalarType::Float));
  for (int i = 0; i < N; i++)
  {
    int pos = *(idx.data_ptr<int>() + i);
    if (*(counts.data_ptr<int>() + pos) == 0)
      continue;
#pragma omp parallel for
    for (int j = 0; j < c; j++)
    {
      *(out.data_ptr<float>() + pos * c + j) += *(inputs.data_ptr<float>() + i * c + j) / (float)(*(counts.data_ptr<int>() + pos));
    }
  }
  return out;
}

at::Tensor cpu_insertion_backward(
    const at::Tensor top_grad,
    const at::Tensor idx,
    const at::Tensor counts,
    const int N)
{
  //return group_point_forward_gpu(points, indices);

  int c = top_grad.size(1);
  int N1 = counts.size(0);
  at::Tensor bottom_grad = torch::zeros({N, c}, at::device(idx.device()).dtype(at::ScalarType::Float));
  for (int i = 0; i < N; i++)
  {
    if (*(counts.data_ptr<int>() + *(idx.data_ptr<int>() + i)) == 0)
      continue;
#pragma omp parallel for
    for (int j = 0; j < c; j++)
    {
      *(bottom_grad.data_ptr<float>() + i * c + j) = *(top_grad.data_ptr<float>() + *(idx.data_ptr<int>() + i) * c + j) / (float)(*(counts.data_ptr<int>() + *(idx.data_ptr<int>() + i)));
    }
  }
  return bottom_grad;
}

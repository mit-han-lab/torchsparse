#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCBlas.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h> 
#include <chrono>
#include <algorithm>
#include "convolution_gpu.h"


void ConvolutionForwardGPU(at::Tensor in_feat, at::Tensor out_feat,
                           at::Tensor kernel, at::Tensor neighbor_map, 
                           at::Tensor neighbor_offset, const bool transpose) {
    
  if (in_feat.size(1) != kernel.size(1)) {
    throw std::invalid_argument("Input feature size and kernel size mismatch");
  }

  int out_nrows = out_feat.size(0);
  out_feat.resize_({out_nrows, kernel.size(2)});
  out_feat.zero_();
  
  
  int kernel_volume = kernel.size(0);
  
  cublasHandle_t handle =
      //THCState_getCurrentBlasHandle(at::globalContext().getTHCState());
      at::cuda::getCurrentCUDABlasHandle();

  ConvolutionForwardKernelGPU(
      in_feat.data_ptr<float>(), in_feat.size(1), out_feat.data_ptr<float>(),
      out_feat.size(1), kernel.data_ptr<float>(), neighbor_map.data_ptr<int>(), 
      neighbor_offset.data_ptr<int>(), in_feat.size(0), out_feat.size(0), 
      kernel.size(0), transpose, handle, 
      at::cuda::getCurrentCUDAStream());
  
   

}

void ConvolutionBackwardGPU(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor kernel, at::Tensor grad_kernel, at::Tensor neighbor_map,
    at::Tensor neighbor_offset, const bool transpose) {
  
  grad_in_feat.resize_as_(in_feat);
  grad_in_feat.zero_();
  grad_kernel.resize_as_(kernel);
  grad_kernel.zero_();
  
  int kernel_volume = kernel.size(0);
  bool flag = false;
  
  cublasHandle_t handle =
      //THCState_getCurrentBlasHandle(at::globalContext().getTHCState());
      at::cuda::getCurrentCUDABlasHandle();
  ConvolutionBackwardKernelGPU(
      in_feat.data_ptr<float>(), grad_in_feat.data_ptr<float>(), in_feat.size(1),
      grad_out_feat.data_ptr<float>(), grad_out_feat.size(1), kernel.data_ptr<float>(),
      grad_kernel.data_ptr<float>(), neighbor_map.data_ptr<int>(), neighbor_offset.data_ptr<int>(), 
      in_feat.size(0), grad_out_feat.size(0), kernel.size(0), 
      transpose, handle, at::cuda::getCurrentCUDAStream());
  
}


/*
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sparseconv_forward", &ConvolutionForwardGPU, "point cloud convolution forward (CUDA)");
  m.def("sparseconv_backward", &ConvolutionBackwardGPU, "point cloud convolution backward (CUDA)");
}
*/

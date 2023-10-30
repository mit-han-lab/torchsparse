#include <torch/torch.h>
#include <torch/extension.h>

#include "exclusive_scan_cuda.h"

// to derive quantified address of activated features
__global__ void exclusive_scan_for_kernel_quantified(
                const int kv, 
                const int *input, 
                const int q, 
                // const int mid_kernel, 
                int *output,
                int *qoutput
                // bool precompute_mid
){
  // a thread for a scan
  const int id = threadIdx.x + 1;
  if (id >= kv){return;}
  int acc = 0;
  int qacc = 0;
#pragma unroll 
  for (int i = 0; i < id; i++){ 
    // if (precompute_mid && i == mid_kernel){continue;}
    acc += input[i];
    qacc += (input[i] + q - 1) / q * q;
  }
  output[id] = acc;
  qoutput[id] = qacc;
}

at::Tensor exclusive_scan_quantified_wrapper(
    const int k_vol, at::Tensor neighbor_offset, 
    at::Tensor neighbor_address, at::Tensor q_neighbor_address){

  int *knnz_ptr = neighbor_offset.data_ptr<int>();
  int *kpos_ptr = neighbor_address.data_ptr<int>();
  int *qkpos_ptr = q_neighbor_address.data_ptr<int>();

  exclusive_scan_for_kernel_quantified<<<1, k_vol, 0, 0>>>(
        k_vol + 1, knnz_ptr, 128, kpos_ptr, qkpos_ptr
  );
  // We must have a tensor as return val for Pybind.
  at::Tensor status = at::zeros({1});
  return status;
}
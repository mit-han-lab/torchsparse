/*
Please consider citing the following paper when using the code:

@inproceedings{hong2023pcengine,
  title={{Exploiting Hardware Utilization and Adaptive Dataflow for Efficient Sparse Convolution in 3D Point Clouds}},
  author={Hong, Ke and Yu, Zhongming and Dai, Guohao and Yang, Xinhao and Lian, Yaoxiu and Liu, Zehao and Xu, Ningyi and Wang, Yu},
  booktitle={Sixth Conference on Machine Learning and Systems (MLSys)},
  year={2023}
}
*/

#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <torch/extension.h>
#include <mma.h>
#if __CUDA_ARCH__ >= 700
#include <cuda/pipeline>
#endif


#include "convolution_forward_fetch_on_demand_cuda.h"

#define DIV_UP(x, y) ((x) + (y) - 1) / (y)

// kernels employed in PCEngine [Fetch-on-Demand]
// device function to indicate the weight index in fetch-on-demand gemms
__device__ __forceinline__ int binary_search(
                            const int *S_csrRowPtr, const int eid, 
                            const int start, const int end) {
    
  int lo = start, hi = end;
  if (lo == hi){
    return lo;
  }
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (__ldg(S_csrRowPtr + mid) <= eid) {
        lo = mid + 1;
    } else {
        hi = mid;
    }
  }
  if (__ldg(S_csrRowPtr + hi) <= eid) {
    return hi;
  } else {
      return hi - 1;
  }
}


/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, blockDim.x = 8, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void fetch_on_demand_gemm_fp32(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f, 
                const float *__restrict__ kw, 
                float *out_f,
                const int *imap, 
                const int *omap) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 2;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const float *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty 
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub[N_LOOP][4] = {0.0f};
  float padding[4] = {0.0f};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ float As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];
  
  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((float4*)(&Bs[ty][ctx])) = ((s + ty) < c_in && cx < c_out) ? 
      *((float4*)(kw_ptr + c_out * (s + ty) + cx)) : 
      *((float4*)(&padding[0]));
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      *((float4*)(&As[n][ty][ctx])) = ((s + ctx) < c_in && in_row > -1) ? 
        *((float4*)(&in_f[c_in * in_row + s + ctx])) : 
        *((float4*)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll 
    for (int n = 0; n < N_LOOP; n++){
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
        float Ast = As[n][ty][k];
#pragma unroll
        for (int c = 0; c < 4; c++){
          Csub[n][c] += Ast * Bs[k][ctx + c];
        }
      }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], Csub[n][c]);
      }
    }
  }
}


/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, blockDim.x = 8, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void fetch_on_demand_gemm_no_fusion_fp32(
                const int knnz,
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f, 
                const float *__restrict__ kw, 
                float *out_f,
                const int *imap, 
                const int *omap) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 2;

  // Weight index
  // const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  // const float *kw_ptr = &kw[widx * c_in * c_out];
  const float *kw_ptr = &kw[0];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub[N_LOOP][4] = {0.0f};
  float padding[4] = {0.0f};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ float As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];
  
  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((float4*)(&Bs[ty][ctx])) = ((s + ty) < c_in && cx < c_out) ? 
      *((float4*)(kw_ptr + c_out * (s + ty) + cx)) : 
      *((float4*)(&padding[0]));
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < knnz ? imap[y_temp] : -1;

      *((float4*)(&As[n][ty][ctx])) = ((s + ctx) < c_in && in_row > -1) ? 
        *((float4*)(&in_f[c_in * in_row + s + ctx])) : 
        *((float4*)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll 
    for (int n = 0; n < N_LOOP; n++){
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
        float Ast = As[n][ty][k];
#pragma unroll
        for (int c = 0; c < 4; c++){
          Csub[n][c] += Ast * Bs[k][ctx + c];
        }
      }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < knnz ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        // atomicAdd(&out_f[c_out * out_row + cx + c], Csub[n][c]);
        out_f[c_out * out_row + cx + c] += Csub[n][c];
      }
    }
  }
}


/*
BLOCK_SIZE = 16, N_LOOP = 8, SKEW = 8, blockDim.x = 4, blockDim.y = 16
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void fetch_on_demand_gemm_fp32_once(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f, 
                const float *__restrict__ kw, 
                float *out_f,
                const int *imap, 
                const int *omap) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 2;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const float *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty 
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub[N_LOOP][4] = {0.0f};
  float padding[4] = {0.0f};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ float As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];
  
  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  // In "loop once" version, s = 0
  // for (int s = 0; s < c_in; s += BLOCK_SIZE) {

  // Load the matrices from device memory
  // to shared memory; each thread loads
  // one element of each matrix

  // Kernel weight to Bs
  *((float4*)(&Bs[ty][ctx])) = ((ty) < c_in && cx < c_out) ? 
    *((float4*)(kw_ptr + c_out * (ty) + cx)) : 
    *((float4*)(&padding[0]));
    
  // Input feature to As
  for (int n = 0; n < N_LOOP; n++){

    int y_temp = y + n * BLOCK_SIZE;

    // The thread deals with the x-th channel of the y-th output
    int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

    *((float4*)(&As[n][ty][ctx])) = ((ctx) < c_in && in_row > -1) ? 
      *((float4*)(&in_f[c_in * in_row + ctx])) : 
      *((float4*)(&padding[0]));
  }

  // Synchronize to make sure the matrices are loaded
  __syncthreads();

  // Multiply the two matrices together;
  // each thread computes one element
  // of the block sub-matrix
#pragma unroll 
  for (int n = 0; n < N_LOOP; n++){
#pragma unroll
    for (int k = 0; k < c_in; ++k) {
      float Ast = As[n][ty][k];
#pragma unroll
      for (int c = 0; c < 4; c++){
        Csub[n][c] += Ast * Bs[k][ctx + c];
      }
    }
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], Csub[n][c]);
      }
    }
  }
}

/*
BLOCK_SIZE = 16, N_LOOP = 8, SKEW = 8, 
blockDim.x = 8, blockDim.y = 16
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void fetch_on_demand_gemm_fp32_2(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f, 
                const float *__restrict__ kw, 
                float *out_f,
                const int *imap, 
                const int *omap) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 1;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const float *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty 
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub[N_LOOP][2] = {0.0f};
  float padding[2] = {0.0f};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ float As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];
  
  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((float2*)(&Bs[ty][ctx])) = ((s + ty) < c_in && cx < c_out) ? 
      *((float2*)(kw_ptr + c_out * (s + ty) + cx)) : 
      *((float2*)(&padding[0]));
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      *((float2*)(&As[n][ty][ctx])) = ((s + ctx) < c_in && in_row > -1) ? 
        *((float2*)(&in_f[c_in * in_row + s + ctx])) : 
        *((float2*)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll 
    for (int n = 0; n < N_LOOP; n++){
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
        float Ast = As[n][ty][k];
#pragma unroll
        for (int c = 0; c < 2; c++){
          Csub[n][c] += Ast * Bs[k][ctx + c];
        }
      }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 2; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], Csub[n][c]);
      }
    }
  }
}


/*
BLOCK_SIZE = 16, N_LOOP = 4, SKEW = 8, 
blockDim.x = 16, blockDim.y = 16
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void fetch_on_demand_gemm_fp32_1(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f, 
                const float *__restrict__ kw, 
                float *out_f,
                const int *imap, 
                const int *omap) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  // const int ctx = tx;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const float *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + tx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty 
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub[N_LOOP] = {0.0f};
  float padding = 0.0f;

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ float As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];
  
  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    Bs[ty][tx] = ((s + ty) < c_in && cx < c_out) ? 
      *(kw_ptr + c_out * (s + ty) + cx) : 
      padding;
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      As[n][ty][tx] = ((s + tx) < c_in && in_row > -1) ? 
        in_f[c_in * in_row + s + tx] : 
        padding;
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll 
    for (int n = 0; n < N_LOOP; n++){
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
        // float Ast = As[n][ty][k];
        // for (int c = 0; c < 2; c++){
        Csub[n] += As[n][ty][k] * Bs[k][tx];
        // }
      }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
      // for (int c = 0; c < 2; c++){
      atomicAdd(&out_f[c_out * out_row + cx], Csub[n]);
      // }
    }
  }
}


/*
BLOCK_SIZE = 16, N_LOOP = 4, SKEW = 8, 
blockDim.x = 16, blockDim.y = 16
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void fetch_on_demand_gemm_no_fusion_fp32_1(
                const int knnz,
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f, 
                const float *__restrict__ kw, 
                float *out_f,
                const int *imap, 
                const int *omap) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  // const int ctx = tx;

  // Weight index
  // const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const float *kw_ptr = &kw[0];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + tx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub[N_LOOP] = {0.0f};
  float padding = 0.0f;

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ float As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];
  
  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    Bs[ty][tx] = ((s + ty) < c_in && cx < c_out) ? 
      *(kw_ptr + c_out * (s + ty) + cx) : 
      padding;
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < knnz ? imap[y_temp] : -1;

      As[n][ty][tx] = ((s + tx) < c_in && in_row > -1) ? 
        in_f[c_in * in_row + s + tx] : 
        padding;
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll 
    for (int n = 0; n < N_LOOP; n++){
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
        // float Ast = As[n][ty][k];
        // for (int c = 0; c < 2; c++){
        Csub[n] += As[n][ty][k] * Bs[k][tx];
        // }
      }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < knnz ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
      // for (int c = 0; c < 2; c++){
      // atomicAdd(&out_f[c_out * out_row + cx], Csub[n]);
      out_f[c_out * out_row + cx] += Csub[n];
      // }
    }
  }
}


/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, blockDim.x = 8, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void fetch_on_demand_gemm_fp16_4_once(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const half *__restrict__ in_f, 
                const half *__restrict__ kw, 
                half *out_f,
                const int *imap, 
                const int *omap) {
# if __CUDA_ARCH__ >= 700
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 2;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const half *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty 
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  half Csub[N_LOOP][4] = {__float2half(0.0f)};
  half padding[4] = {__float2half(0.0f)};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ half As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];
  
  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  // In "loop once" version, s = 0
  // for (int s = 0; s < c_in; s += BLOCK_SIZE) {

  // Kernel weight to Bs
  *((float2*)(&Bs[ty][ctx])) = (ty < c_in && cx < c_out) ? 
    *((float2*)(kw_ptr + c_out * ty + cx)) : 
    *((float2*)(&padding[0]));
    
  int y_temp = y;
  // Input feature to As
  for (int n = 0; n < N_LOOP; n++){

    // The thread deals with the x-th channel of the y-th output
    int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

    *((float2*)(&As[n][ty][ctx])) = (ctx < c_in && in_row > -1) ? 
      *((float2*)(&in_f[c_in * in_row + ctx])) : 
      *((float2*)(&padding[0]));
      
    y_temp += BLOCK_SIZE;
  }

  // Synchronize to make sure the matrices are loaded
  __syncthreads();

  // Multiply the two matrices together;
  // each thread computes one element
  // of the block sub-matrix
#pragma unroll 
  for (int n = 0; n < N_LOOP; n++){
#pragma unroll
    for (int k = 0; k < c_in; ++k){
      half Ast = As[n][ty][k];
#pragma unroll
      for (int c = 0; c < 4; c++){
        Csub[n][c] = __hfma(Ast, Bs[k][ctx + c], Csub[n][c]);
      }
    }

    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], Csub[n][c]);
      }
    }
  }
#else
  #pragma message("FP16 kernels will not be compiled.")
#endif
}


/*
BLOCK_SIZE = 16, N_LOOP = 8, SKEW = 8, blockDim.x = 8, blockDim.y = 16
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void fetch_on_demand_gemm_fp16_2(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const half *__restrict__ in_f, 
                const half *__restrict__ kw, 
                half *out_f,
                const int *imap, 
                const int *omap) {
# if __CUDA_ARCH__ >= 700
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 1;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const half *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty 
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  half Csub[N_LOOP][2] = {__float2half(0.0f)};
  half padding[2] = {__float2half(0.0f)};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ half As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];
  
  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((float*)(&Bs[ty][ctx])) = ((s + ty) < c_in && cx < c_out) ? 
      *((float*)(kw_ptr + c_out * (s + ty) + cx)) : 
      *((float*)(&padding[0]));
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      *((float*)(&As[n][ty][ctx])) = ((s + ctx) < c_in && in_row > -1) ? 
        *((float*)(&in_f[c_in * in_row + s + ctx])) : 
        *((float*)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll 
    for (int n = 0; n < N_LOOP; n++){
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
        half Ast = As[n][ty][k];
#pragma unroll
        for (int c = 0; c < 2; c++){
          Csub[n][c] = __hfma(Ast, Bs[k][ctx + c], Csub[n][c]);
        }
      }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 2; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], Csub[n][c]);
      }
    }
  }
#else
  #pragma message("TF32 kernels will not be compiled.")
#endif
}


/*
BLOCK_SIZE = 16, N_LOOP = 4, SKEW = 8, blockDim.x = 16, blockDim.y = 16
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void fetch_on_demand_gemm_fp16_1(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const half *__restrict__ in_f, 
                const half *__restrict__ kw, 
                half *out_f,
                const int *imap, 
                const int *omap) {
# if __CUDA_ARCH__ >= 700
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  // const int ctx = tx << 1;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const half *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + tx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty 
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  half Csub[N_LOOP] = {__float2half(0.0f)};
  half padding = __float2half(0.0f);

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ half As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];
  
  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    Bs[ty][tx] = ((s + ty) < c_in && cx < c_out) ? 
      *(kw_ptr + c_out * (s + ty) + cx) : 
      padding;
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      As[n][ty][tx] = ((s + tx) < c_in && in_row > -1) ? 
        in_f[c_in * in_row + s + tx] : 
        padding;
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll 
    for (int n = 0; n < N_LOOP; n++){
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
        // half Ast = As[n][ty][k];
        // for (int c = 0; c < 2; c++){
          Csub[n] = __hfma(As[n][ty][k], Bs[k][tx], Csub[n]);
        // }
      }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
      // for (int c = 0; c < 2; c++){
      atomicAdd(&out_f[c_out * out_row + cx], Csub[n]);
      // }
    }
  }
#else
  #pragma message("FP16 kernels will not be compiled.")
#endif
}


/*
BLOCK_SIZE = 16, N_LOOP = 4, SKEW = 8, blockDim.x = 16, blockDim.y = 16
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void fetch_on_demand_gemm_no_fusion_fp16_1(
                const int knnz,
                const int c_in,
                const int c_out,
                const half *__restrict__ in_f, 
                const half *__restrict__ kw, 
                half *out_f,
                const int *imap, 
                const int *omap) {

#if __CUDA_ARCH__ >= 700

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  // const int ctx = tx << 1;

  // Weight index
  // const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const half *kw_ptr = &kw[0];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + tx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  half Csub[N_LOOP] = {__float2half(0.0f)};
  half padding = __float2half(0.0f);

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ half As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];
  
  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    Bs[ty][tx] = ((s + ty) < c_in && cx < c_out) ? 
      *(kw_ptr + c_out * (s + ty) + cx) : 
      padding;
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < knnz ? imap[y_temp] : -1;

      As[n][ty][tx] = ((s + tx) < c_in && in_row > -1) ? 
        in_f[c_in * in_row + s + tx] : 
        padding;
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll 
    for (int n = 0; n < N_LOOP; n++){
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
        // half Ast = As[n][ty][k];
        // for (int c = 0; c < 2; c++){
          Csub[n] = __hfma(As[n][ty][k], Bs[k][tx], Csub[n]);
        // }
      }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < knnz ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
      // for (int c = 0; c < 2; c++){
      // atomicAdd(&out_f[c_out * out_row + cx], Csub[n]);
      out_f[c_out * out_row + cx] = 
        __hadd(out_f[c_out * out_row + cx], Csub[n]);
      // }
    }
  }
#else
  #pragma message("FP16 kernels will not be compiled.")
#endif
}


// kernels using tensor cores
// using namespace nvcuda;
/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, M = 16, K = 8, N = 16, 
MS = 2, NS = 2, WS = 4 = MS x NS
blockDim.x = 8, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW, 
  int M, int K, int N, int WS, int MS, int NS>
__global__ void fetch_on_demand_gemm_tf32(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f, 
                const float *__restrict__ kw, 
                float *out_f,
                const int *imap, 
                const int *omap) {
#if __CUDA_ARCH__ >= 800

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 2;
  const int tid = ty * blockDim.x + tx;

  // Warp index
  const int warpId = tid / 32;
  const int laneId = tid % 32;
  const int warp_row = warpId / NS;
  const int warp_col = warpId % NS;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const float *kw_ptr = &kw[widx * c_in * c_out];
  
  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty 
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  // float Csub[N_LOOP][4] = {0.0f};
  float padding[4] = {0.0f};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ float As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memeory array Cs used to
  // store the sub-matrix of C
  // __shared__ float Cs[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Fragments to store As, Bs and Cs
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, float> c[N_LOOP / 2];

#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    nvcuda::wmma::fill_fragment(c[n], 0.0f);
  }
  
  // May not be necessary
  __syncthreads();

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {
    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((float4*)(&Bs[ty][ctx])) = ((s + ty) < c_in && cx < c_out) ? 
      *((float4*)(kw_ptr + c_out * (s + ty) + cx)) : 
      *((float4*)(&padding[0]));
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      *((float4*)(&As[n][ty][ctx])) = ((s + ctx) < c_in && in_row > -1) ? 
        *((float4*)(&in_f[c_in * in_row + s + ctx])) : 
        *((float4*)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together using Tensor Core
    // Load data from shmem to tensor core
    // Just load Bs once
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k += K){
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M, N, K, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> a[N_LOOP / 2];
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M, N, K, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> b;
      nvcuda::wmma::load_matrix_sync(b, &Bs[k][warp_col * N], BLOCK_SIZE + SKEW);
#pragma unroll
      for (int t = 0; t < b.num_elements; t++) {
          b.x[t] = nvcuda::wmma::__float_to_tf32(b.x[t]);
      }
#pragma unroll
      for (int n = 0; n < N_LOOP / 2; n++){
        nvcuda::wmma::load_matrix_sync(a[n], &As[n * MS + warpId / WS][warp_row % MS * M][k], BLOCK_SIZE + SKEW);
#pragma unroll
        for (int t = 0; t < a[n].num_elements; t++) {
          a[n].x[t] = nvcuda::wmma::__float_to_tf32(a[n].x[t]);
        }
        nvcuda::wmma::mma_sync(c[n], a[n], b, c[n]);
      }  
    }
    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Store C fragments to shared memory
  // Note that we reuse As for Cs storing
#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    nvcuda::wmma::store_matrix_sync(&As[n * MS + warpId / WS][warp_row % MS * M][warp_col * N], 
      c[n], BLOCK_SIZE + SKEW, nvcuda::wmma::mem_row_major);
  }

  // Synchronize to make sure that all C fragments are 
  // stored into shared memory
  __syncthreads();

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], As[n][ty][ctx + c]);
      }
    }
  }
#else
  #pragma message("TF32 kernels will not be compiled.")
#endif
}


/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, M = 16, K = 8, N = 16, 
MS = 2, NS = 2, WS = 4 = MS x NS
blockDim.x = 8, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW, 
  int M, int K, int N, int WS, int MS, int NS>
__global__ void fetch_on_demand_gemm_no_fusion_tf32(
                const int knnz,
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f, 
                const float *__restrict__ kw, 
                float *out_f,
                const int *imap, 
                const int *omap) {
#if __CUDA_ARCH__ >= 800
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 2;
  const int tid = ty * blockDim.x + tx;

  // Warp index
  const int warpId = tid / 32;
  const int laneId = tid % 32;
  const int warp_row = warpId / NS;
  const int warp_col = warpId % NS;

  // Weight index
  // const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  // const float *kw_ptr = &kw[widx * c_in * c_out];
  const float *kw_ptr = &kw[0];
  
  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  // float Csub[N_LOOP][4] = {0.0f};
  float padding[4] = {0.0f};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ float As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memeory array Cs used to
  // store the sub-matrix of C
  // __shared__ float Cs[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Fragments to store As, Bs and Cs
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, float> c[N_LOOP / 2];

#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    nvcuda::wmma::fill_fragment(c[n], 0.0f);
  }
  
  // May not be necessary
  __syncthreads();

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {
    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((float4*)(&Bs[ty][ctx])) = ((s + ty) < c_in && cx < c_out) ? 
      *((float4*)(kw_ptr + c_out * (s + ty) + cx)) : 
      *((float4*)(&padding[0]));
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < knnz ? imap[y_temp] : -1;

      *((float4*)(&As[n][ty][ctx])) = ((s + ctx) < c_in && in_row > -1) ? 
        *((float4*)(&in_f[c_in * in_row + s + ctx])) : 
        *((float4*)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together using Tensor Core
    // Load data from shmem to tensor core
    // Just load Bs once
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k += K){
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M, N, K, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> a[N_LOOP / 2];
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M, N, K, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> b;
      nvcuda::wmma::load_matrix_sync(b, &Bs[k][warp_col * N], BLOCK_SIZE + SKEW);
#pragma unroll
      for (int t = 0; t < b.num_elements; t++) {
          b.x[t] = nvcuda::wmma::__float_to_tf32(b.x[t]);
      }
#pragma unroll
      for (int n = 0; n < N_LOOP / 2; n++){
        nvcuda::wmma::load_matrix_sync(a[n], &As[n * MS + warpId / WS][warp_row % MS * M][k], BLOCK_SIZE + SKEW);
#pragma unroll
        for (int t = 0; t < a[n].num_elements; t++) {
          a[n].x[t] = nvcuda::wmma::__float_to_tf32(a[n].x[t]);
        }
        nvcuda::wmma::mma_sync(c[n], a[n], b, c[n]); 
      }  
    }
    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Store C fragments to shared memory
  // Note that we reuse As for Cs storing
#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    nvcuda::wmma::store_matrix_sync(&As[n * MS + warpId / WS][warp_row % MS * M][warp_col * N], 
      c[n], BLOCK_SIZE + SKEW, nvcuda::wmma::mem_row_major);
  }

  // Synchronize to make sure that all C fragments are 
  // stored into shared memory
  __syncthreads();

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < knnz ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        // atomicAdd(&out_f[c_out * out_row + cx + c], As[n][ty][ctx + c]);
        out_f[c_out * out_row + cx + c] += As[n][ty][ctx + c];
      }
    }
  }
#else
  #pragma message("TF32 kernels will not be compiled.")
#endif
}
////////////////////////////// CUDA_ARCH >= 800 for TF32 ///////////////////////////////////

/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, M = 16, K = 16, N = 16, 
MS = 2, NS = 2, WS = 4 = MS x NS
blockDim.x = 8, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW, 
  int M, int K, int N, int WS, int MS, int NS>
__global__ void fetch_on_demand_gemm_fp16_tc4(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const half *__restrict__ in_f, 
                const half *__restrict__ kw, 
                half *out_f,
                const int *imap, 
                const int *omap) {
#if __CUDA_ARCH__ >= 700

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 2;
  const int tid = ty * blockDim.x + tx;

  // Warp index
  const int warpId = tid / 32;
  // const int laneId = tid % 32;
  const int warp_row = warpId / NS;
  const int warp_col = warpId % NS;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const half *kw_ptr = &kw[widx * c_in * c_out];
  
  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty 
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  // float Csub[N_LOOP][4] = {0.0f};
  half padding[4] = {__float2half(0.0f)};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ half As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memeory array Cs used to
  // store the sub-matrix of C
  // __shared__ float Cs[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Fragments to store As, Bs and Cs
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, half> c[N_LOOP / 2];

#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    nvcuda::wmma::fill_fragment(c[n], __float2half(0.0f));
  }
  
  // May not be necessary
  __syncthreads();

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {
    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((float2*)(&Bs[ty][ctx])) = ((s + ty) < c_in && cx < c_out) ? 
      *((float2*)(kw_ptr + c_out * (s + ty) + cx)) : 
      *((float2*)(&padding[0]));
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      *((float2*)(&As[n][ty][ctx])) = ((s + ctx) < c_in && in_row > -1) ? 
        *((float2*)(&in_f[c_in * in_row + s + ctx])) : 
        *((float2*)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together using Tensor Core
    // Load data from shmem to tensor core
    // Just load Bs once
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k += K){
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M, N, K, half, nvcuda::wmma::row_major> a[N_LOOP / 2];
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M, N, K, half, nvcuda::wmma::row_major> b;
      nvcuda::wmma::load_matrix_sync(b, &Bs[k][warp_col * N], BLOCK_SIZE + SKEW);
#pragma unroll
      for (int n = 0; n < N_LOOP / 2; n++){
        nvcuda::wmma::load_matrix_sync(a[n], &As[n * MS + warpId / WS][warp_row % MS * M][k], BLOCK_SIZE + SKEW);
        nvcuda::wmma::mma_sync(c[n], a[n], b, c[n]);
      }  
    }
    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Store C fragments to shared memory
  // Note that we reuse As for Cs storing
#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    nvcuda::wmma::store_matrix_sync(&As[n * MS + warpId / WS][warp_row % MS * M][warp_col * N], 
      c[n], BLOCK_SIZE + SKEW, nvcuda::wmma::mem_row_major);
  }

  // Synchronize to make sure that all C fragments are 
  // stored into shared memory
  __syncthreads();

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], As[n][ty][ctx + c]);
      }
    }
  }
#else
  #pragma message("FP16 kernels will not be compiled.")
#endif
}


/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, M = 16, K = 16, N = 16, 
MS = 2, NS = 2, WS = 4 = MS x NS
blockDim.x = 8, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW, 
  int M, int K, int N, int WS, int MS, int NS>
__global__ void fetch_on_demand_gemm_fp16_tc4_async(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const half *__restrict__ in_f, 
                const half *__restrict__ kw, 
                half *out_f,
                const int *imap, 
                const int *omap) {
#if __CUDA_ARCH__ >= 700

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 2;
  const int tid = ty * blockDim.x + tx;

  // Warp index
  const int warpId = tid / 32;
  // const int laneId = tid % 32;
  const int warp_row = warpId / NS;
  const int warp_col = warpId % NS;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const half *kw_ptr = &kw[widx * c_in * c_out];
  
  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty 
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  // float Csub[N_LOOP][4] = {0.0f};
  half padding[4] = {__float2half(0.0f)};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ half As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memeory array Cs used to
  // store the sub-matrix of C
  // __shared__ float Cs[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Pipelined copy between gmem and shmem
  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
  const auto shape4 = cuda::aligned_size_t<alignof(float2)>(sizeof(float2));

  // Fragments to store As, Bs and Cs
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, half> c[N_LOOP / 2];

#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    nvcuda::wmma::fill_fragment(c[n], __float2half(0.0f));
  }
  
  // May not be necessary
  __syncthreads();

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {
    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    // const half *kw2Bs_ptr = ((s + ty) < c_in && cx < c_out) ? 
    //   kw_ptr + c_out * (s + ty) + cx : &padding[0];
    pipe.producer_acquire();
    if ((s + ty) < c_in && cx < c_out){
      cuda::memcpy_async(&Bs[ty][ctx], kw_ptr + c_out * (s + ty) + cx, shape4, pipe);
    }
    else{
      cuda::memcpy_async(&Bs[ty][ctx], &padding[0], shape4, pipe);
    }
    // cuda::memcpy_async(&Bs[ty][ctx], kw2Bs_ptr, shape4, pipe);
    pipe.producer_commit();
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      // const half *inf2As_ptr = ((s + ctx) < c_in && in_row > -1) ? 
      //   &in_f[c_in * in_row + s + ctx] : &padding[0];
      pipe.producer_acquire();
      if ((s + ctx) < c_in && in_row > -1){
        cuda::memcpy_async(&As[n][ty][ctx], &in_f[c_in * in_row + s + ctx], shape4, pipe);
      }
      else{
        cuda::memcpy_async(&As[n][ty][ctx], &padding[0], shape4, pipe);
      }
      // cuda::memcpy_async(&As[n][ty][ctx], inf2As_ptr, shape4, pipe);
      pipe.producer_commit();
    }

    // Synchronize to make sure the matrices are loaded
    cuda::pipeline_consumer_wait_prior<0>(pipe);
    __syncthreads();

    // Multiply the two matrices together using Tensor Core
    // Load data from shmem to tensor core
    // Just load Bs once
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k += K){
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M, N, K, half, nvcuda::wmma::row_major> a[N_LOOP / 2];
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M, N, K, half, nvcuda::wmma::row_major> b;
      nvcuda::wmma::load_matrix_sync(b, &Bs[k][warp_col * N], BLOCK_SIZE + SKEW);
#pragma unroll
      for (int n = 0; n < N_LOOP / 2; n++){
        nvcuda::wmma::load_matrix_sync(a[n], &As[n * MS + warpId / WS][warp_row % MS * M][k], BLOCK_SIZE + SKEW);
      }  
#pragma unroll 
      for (int n = 0; n < N_LOOP / 2; n++){
        nvcuda::wmma::mma_sync(c[n], a[n], b, c[n]);
      }
    }
    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    pipe.consumer_release();
    __syncthreads();
  }

  // Store C fragments to shared memory
  // Note that we reuse As for Cs storing
#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    nvcuda::wmma::store_matrix_sync(&As[n * MS + warpId / WS][warp_row % MS * M][warp_col * N], 
      c[n], BLOCK_SIZE + SKEW, nvcuda::wmma::mem_row_major);
  }

  // Synchronize to make sure that all C fragments are 
  // stored into shared memory
  __syncthreads();

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], As[n][ty][ctx + c]);
      }
    }
  }
#else
  #pragma message("FP16 kernels with asynchronous copy will not be compiled.")
#endif
}


/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, M = 16, K = 16, N = 16, 
MS = 2, NS = 2, WS = 4 = MS x NS
blockDim.x = 8, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW, 
  int M, int K, int N, int WS, int MS, int NS>
__global__ void fetch_on_demand_gemm_no_fusion_fp16(
                const int knnz,
                const int c_in,
                const int c_out,
                const half *__restrict__ in_f, 
                const half *__restrict__ kw, 
                half *out_f,
                const int *imap, 
                const int *omap) {
#if __CUDA_ARCH__ >= 700
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 2;
  const int tid = ty * blockDim.x + tx;

  // Warp index
  const int warpId = tid / 32;
  // const int laneId = tid % 32;
  const int warp_row = warpId / NS;
  const int warp_col = warpId % NS;

  // Weight index
  // const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  // const half *kw_ptr = &kw[widx * c_in * c_out];
  const half *kw_ptr = &kw[0];
  
  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  // float Csub[N_LOOP][4] = {0.0f};
  half padding[4] = {__float2half(0.0f)};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ half As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memeory array Cs used to
  // store the sub-matrix of C
  // __shared__ float Cs[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Fragments to store As, Bs and Cs
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, half> c[N_LOOP / 2];

#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    nvcuda::wmma::fill_fragment(c[n], __float2half(0.0f));
  }
  
  // May not be necessary
  __syncthreads();

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {
    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((float2*)(&Bs[ty][ctx])) = ((s + ty) < c_in && cx < c_out) ? 
      *((float2*)(kw_ptr + c_out * (s + ty) + cx)) : 
      *((float2*)(&padding[0]));
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < knnz ? imap[y_temp] : -1;

      *((float2*)(&As[n][ty][ctx])) = ((s + ctx) < c_in && in_row > -1) ? 
        *((float2*)(&in_f[c_in * in_row + s + ctx])) : 
        *((float2*)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together using Tensor Core
    // Load data from shmem to tensor core
    // Just load Bs once
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k += K){
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M, N, K, half, nvcuda::wmma::row_major> a[N_LOOP / 2];
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M, N, K, half, nvcuda::wmma::row_major> b;
      nvcuda::wmma::load_matrix_sync(b, &Bs[k][warp_col * N], BLOCK_SIZE + SKEW);
#pragma unroll
      for (int n = 0; n < N_LOOP / 2; n++){
        nvcuda::wmma::load_matrix_sync(a[n], &As[n * MS + warpId / WS][warp_row % MS * M][k], BLOCK_SIZE + SKEW);
        nvcuda::wmma::mma_sync(c[n], a[n], b, c[n]);
      }  
    }
    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Store C fragments to shared memory
  // Note that we reuse As for Cs storing
#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    nvcuda::wmma::store_matrix_sync(&As[n * MS + warpId / WS][warp_row % MS * M][warp_col * N], 
      c[n], BLOCK_SIZE + SKEW, nvcuda::wmma::mem_row_major);
  }

  // Synchronize to make sure that all C fragments are 
  // stored into shared memory
  __syncthreads();

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < knnz ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        // out_f[c_out * out_row + cx + c] += As[n][ty][ctx + c];
        out_f[c_out * out_row + cx + c] = 
          __hadd(out_f[c_out * out_row + cx + c], As[n][ty][ctx + c]);
      }
    }
  }
#else
  #pragma message("FP16 kernels will not be compiled.")
#endif
}
///////////////////////////////// CUDA_ARCH >= 700 ///////////////////////////////////




// in_feat: (N, c) N=# of input points, c = input channels
// out_feat: (M, o) M=# of output points, o = output channels
//                  for stride=1, M=N. For stride>1, the N input coords
//                  are requantized to M points with grid size (stride *
//                  cur_stride)
// kernel: (k^3, c, o) for a 3D convolution of length k
// neighbor_map: (a, 2) the hash table query results from in_coords to
//                      out_coords
//                      where neighbor_map[:,0] is the index of the input
//                      feature and neighbor_map[:,1] is the index of the output
//                      feature
// neighbor_offset: (k^3) count of active weights based on neighbor_map
//                      with unused weights having 0 and neighbor_offset[k^3/2]
//                      holding w[0,0].
at::Tensor conv_forward_fetch_on_demand_cuda(
    at::Tensor in_feat, at::Tensor kernel, 
    at::Tensor neighbor_map, const int sum_nnz, 
    at::Tensor neighbor_address, at::Tensor q_neighbor_address,
    const int output_size, const int qsum_nnz, const bool transpose, 
    const bool allow_tf32, const bool allow_fp16) {

  // int sum_nnz = (int)torch::sum(neighbor_offset).item<int>();
  int input_size = in_feat.size(0);
  int in_channel = in_feat.size(1);
  int out_channel = kernel.size(2);
  int k_vol = kernel.size(0);
  // int *knnz_ptr = neighbor_offset.data_ptr<int>();
  // int *in_map_ptr = in_neighbor_map.data_ptr<int>();
  // int *out_map_ptr = out_neighbor_map.data_ptr<int>();
  int *kpos_ptr = neighbor_address.data_ptr<int>();
  int *qkpos_ptr = q_neighbor_address.data_ptr<int>();
  int *in_map_ptr;
  int *out_map_ptr;
  if (transpose){
    in_map_ptr = neighbor_map.data_ptr<int>() + sum_nnz;
    out_map_ptr = neighbor_map.data_ptr<int>();
  }
  else{
    in_map_ptr = neighbor_map.data_ptr<int>();
    out_map_ptr = neighbor_map.data_ptr<int>() + sum_nnz;
  }

  // memory allocation
  at::Tensor out_feat = torch::zeros({output_size, out_channel}, 
            at::device(in_feat.device()).dtype(in_feat.scalar_type()));
  // at::Tensor kpos = torch::zeros({k_vol + 1}, 
  //           at::device(in_feat.device()).dtype(at::ScalarType::Int));
  // at::Tensor qkpos = torch::zeros({k_vol + 1}, 
  //           at::device(in_feat.device()).dtype(at::ScalarType::Int));
  // int *kpos_ptr = kpos.data_ptr<int>();
  // int *qkpos_ptr = qkpos.data_ptr<int>();

  // should be modified in the future
  int mid_kernel = (k_vol % 2 == 1) ? k_vol / 2 : 0;

  bool data_type_half = in_feat.scalar_type() == at::ScalarType::Half;
  // bool precompute_mid = (input_size == output_size && k_vol % 2 == 1);
  bool precompute_mid = false;

  // exclusive_scan_for_kernel_quantified<<<1, k_vol, 0, 0>>>(
  //       k_vol + 1, knnz_ptr, 128, kpos_ptr, qkpos_ptr
  // );

  // int qsum_nnz = qkpos[k_vol].item<int>();
  // printf("%d", qsum_nnz);

  if (data_type_half && allow_fp16){
    if (in_channel % 4 == 0 && out_channel % 4 == 0){    
      if (in_channel <= 16 || out_channel <= 16){
        fetch_on_demand_gemm_fp16_4_once<16, 4, 8>
                    <<<dim3(DIV_UP(out_channel, 16), DIV_UP(qsum_nnz, 64), 1), dim3(4, 16, 1)>>>(
                    kpos_ptr, qkpos_ptr, k_vol, in_channel, out_channel, 
                    reinterpret_cast<half *>(in_feat.data_ptr<at::Half>()),
                    reinterpret_cast<half *>(kernel.data_ptr<at::Half>()),
                    reinterpret_cast<half *>(out_feat.data_ptr<at::Half>()),
                    in_map_ptr, out_map_ptr);
      }
      else{
        if (allow_tf32){
          fetch_on_demand_gemm_fp16_tc4_async<32, 4, 8, 16, 16, 16, 4, 2, 2>
                    <<<dim3(DIV_UP(out_channel, 32), DIV_UP(qsum_nnz, 128), 1), dim3(8, 32, 1)>>>(
                    kpos_ptr, qkpos_ptr, k_vol, in_channel, out_channel, 
                    reinterpret_cast<half *>(in_feat.data_ptr<at::Half>()),
                    reinterpret_cast<half *>(kernel.data_ptr<at::Half>()),
                    reinterpret_cast<half *>(out_feat.data_ptr<at::Half>()),
                    in_map_ptr, out_map_ptr);
        }
        else{
          fetch_on_demand_gemm_fp16_tc4<32, 4, 8, 16, 16, 16, 4, 2, 2>
                    <<<dim3(DIV_UP(out_channel, 32), DIV_UP(qsum_nnz, 128), 1), dim3(8, 32, 1)>>>(
                    kpos_ptr, qkpos_ptr, k_vol, in_channel, out_channel, 
                    reinterpret_cast<half *>(in_feat.data_ptr<at::Half>()),
                    reinterpret_cast<half *>(kernel.data_ptr<at::Half>()),
                    reinterpret_cast<half *>(out_feat.data_ptr<at::Half>()),
                    in_map_ptr, out_map_ptr);
        }
      }
    }
    else if (in_channel % 2 == 0 && out_channel % 2 == 0){
        fetch_on_demand_gemm_fp16_2<16, 8, 8>
                    <<<dim3(DIV_UP(out_channel, 16), DIV_UP(qsum_nnz, 128), 1), dim3(8, 16, 1)>>>(
                    kpos_ptr, qkpos_ptr, k_vol, in_channel, out_channel, 
                    reinterpret_cast<half *>(in_feat.data_ptr<at::Half>()),
                    reinterpret_cast<half *>(kernel.data_ptr<at::Half>()),
                    reinterpret_cast<half *>(out_feat.data_ptr<at::Half>()),
                    in_map_ptr, out_map_ptr);   
    }
    else{
        fetch_on_demand_gemm_fp16_1<16, 4, 8>
                    <<<dim3(DIV_UP(out_channel, 16), DIV_UP(qsum_nnz, 64), 1), dim3(16, 16, 1)>>>(
                    kpos_ptr, qkpos_ptr, k_vol, in_channel, out_channel, 
                    reinterpret_cast<half *>(in_feat.data_ptr<at::Half>()),
                    reinterpret_cast<half *>(kernel.data_ptr<at::Half>()),
                    reinterpret_cast<half *>(out_feat.data_ptr<at::Half>()),
                    in_map_ptr, out_map_ptr);  
    }  
  }
  else{
    if(in_channel % 4 == 0 && out_channel % 4 ==0){
      if (in_channel <= 16 && out_channel <= 16){
        fetch_on_demand_gemm_fp32_once<16, 4, 8>
                    <<<dim3(DIV_UP(out_channel, 16), DIV_UP(qsum_nnz, 64), 1), dim3(4, 16, 1)>>>(
                    kpos_ptr, qkpos_ptr, k_vol, in_channel, out_channel, 
                    in_feat.data_ptr<float>(), kernel.data_ptr<float>(), out_feat.data_ptr<float>(), 
                    in_map_ptr, out_map_ptr);
      }
      else{
        if (allow_tf32){
            fetch_on_demand_gemm_tf32<32, 4, 8, 16, 8, 16, 4, 2, 2>
                    <<<dim3(DIV_UP(out_channel, 32), DIV_UP(qsum_nnz, 128), 1), dim3(8, 32, 1)>>>(
                    kpos_ptr, qkpos_ptr, k_vol, in_channel, out_channel, 
                    in_feat.data_ptr<float>(), kernel.data_ptr<float>(), out_feat.data_ptr<float>(), 
                    in_map_ptr, out_map_ptr);
        }
        else{
            fetch_on_demand_gemm_fp32<32, 4, 8>
                    <<<dim3(DIV_UP(out_channel, 32), DIV_UP(qsum_nnz, 128), 1), dim3(8, 32, 1)>>>(
                    kpos_ptr, qkpos_ptr, k_vol, in_channel, out_channel, 
                    in_feat.data_ptr<float>(), kernel.data_ptr<float>(), out_feat.data_ptr<float>(), 
                    in_map_ptr, out_map_ptr);
        }
      }
    }
    else if (in_channel % 2 == 0 && out_channel % 2 == 0){
        fetch_on_demand_gemm_fp32_2<16, 8, 8>
                    <<<dim3(DIV_UP(out_channel, 16), DIV_UP(qsum_nnz, 128), 1), dim3(8, 16, 1)>>>(
                    kpos_ptr, qkpos_ptr, k_vol, in_channel, out_channel, 
                    in_feat.data_ptr<float>(), kernel.data_ptr<float>(), out_feat.data_ptr<float>(), 
                    in_map_ptr, out_map_ptr);
    }
    else{
        fetch_on_demand_gemm_fp32_1<16, 4, 8>
                    <<<dim3(DIV_UP(out_channel, 16), DIV_UP(qsum_nnz, 64), 1), dim3(16, 16, 1)>>>(
                    kpos_ptr, qkpos_ptr, k_vol, in_channel, out_channel, 
                    in_feat.data_ptr<float>(), kernel.data_ptr<float>(), out_feat.data_ptr<float>(), 
                    in_map_ptr, out_map_ptr);
    }
  }

  // precomputation only for odd channel size
  // bool precompute_mid = (input_size == output_size && k_vol % 2 == 1);
  if (precompute_mid){
    at::addmm_out(out_feat, out_feat, in_feat, kernel[mid_kernel]);
  }

  return out_feat;
}


at::Tensor conv_forward_fetch_on_demand_no_fusion_cuda(
    at::Tensor in_feat, at::Tensor kernel,
    at::Tensor neighbor_map, at::Tensor neighbor_offset, 
    const int sum_nnz, const int output_size, const bool transpose, 
    const bool allow_tf32, const bool allow_fp16){

  // int sum_nnz = (int)torch::sum(neighbor_offset).item<int>();
  int input_size = in_feat.size(0);
  int in_channel = in_feat.size(1);
  int out_channel = kernel.size(2);
  int k_vol = kernel.size(0);
  int *knnz_ptr = neighbor_offset.data_ptr<int>();
  // int *in_map_ptr = in_neighbor_map.data_ptr<int>();
  // int *out_map_ptr = out_neighbor_map.data_ptr<int>();
  // int *kpos_ptr = neighbor_address.data_ptr<int>();
  // int *qkpos_ptr = q_neighbor_address.data_ptr<int>();
  int *in_map_ptr;
  int *out_map_ptr;
  if (transpose){
    in_map_ptr = neighbor_map.data_ptr<int>() + sum_nnz;
    out_map_ptr = neighbor_map.data_ptr<int>();
  }
  else{
    in_map_ptr = neighbor_map.data_ptr<int>();
    out_map_ptr = neighbor_map.data_ptr<int>() + sum_nnz;
  }

  // memory allocation
  at::Tensor out_feat = torch::zeros({output_size, out_channel}, 
            at::device(in_feat.device()).dtype(in_feat.scalar_type()));
  // at::Tensor kpos = torch::zeros({k_vol + 1}, 
  //           at::device(in_feat.device()).dtype(at::ScalarType::Int));
  // at::Tensor qkpos = torch::zeros({k_vol + 1}, 
  //           at::device(in_feat.device()).dtype(at::ScalarType::Int));
  // int *kpos_ptr = kpos.data_ptr<int>();
  // int *qkpos_ptr = qkpos.data_ptr<int>();

  // should be modified in the future
  int mid_kernel = (k_vol % 2 == 1) ? k_vol / 2 : 0;

  bool data_type_half = in_feat.scalar_type() == at::ScalarType::Half;
  // bool precompute_mid = (input_size == output_size && k_vol % 2 == 1);
  bool precompute_mid = false;

  /********************************************************************/
  // loop over all kernel offsets
  int cur_idx = 0;
  // int stream_id = 0;
  for (int k = 0; k < k_vol; k++){
    int cur_nnz = knnz_ptr[k];
    
    if (cur_nnz == 0){continue;}

    // size_t gridnum_x = DIV_UP(out_channel, 32);
    // size_t gridnum_y = DIV_UP(cur_nnz, 32);

    if (data_type_half && allow_fp16){
      if (in_channel % 4 == 0 && out_channel % 4 == 0){
        fetch_on_demand_gemm_no_fusion_fp16<32, 4, 8, 16, 16, 16, 4, 2, 2>
              <<<dim3(DIV_UP(out_channel, 32), DIV_UP(cur_nnz, 32), 1), dim3(8, 32, 1)>>>(
                    cur_nnz, in_channel, out_channel, 
                    reinterpret_cast<half *>(in_feat.data_ptr<at::Half>()), 
                    reinterpret_cast<half *>(kernel.data_ptr<at::Half>() 
                        + k * in_channel * out_channel), 
                    reinterpret_cast<half *>(out_feat.data_ptr<at::Half>()), 
                    &in_map_ptr[cur_idx], &out_map_ptr[cur_idx]
                );
      }
      else{
        fetch_on_demand_gemm_no_fusion_fp16_1<16, 4, 8>
              <<<dim3(DIV_UP(out_channel, 16), DIV_UP(cur_nnz, 16), 1), dim3(16, 16, 1)>>>(
                    cur_nnz, in_channel, out_channel, 
                    reinterpret_cast<half *>(in_feat.data_ptr<at::Half>()), 
                    reinterpret_cast<half *>(kernel.data_ptr<at::Half>() 
                        + k * in_channel * out_channel), 
                    reinterpret_cast<half *>(out_feat.data_ptr<at::Half>()), 
                    &in_map_ptr[cur_idx], &out_map_ptr[cur_idx]
                );
      }
    }
    else{
      if (in_channel % 4 == 0 && out_channel % 4 == 0){
        if (allow_tf32){
          fetch_on_demand_gemm_no_fusion_tf32<32, 4, 8, 16, 8, 16, 4, 2, 2>
              <<<dim3(DIV_UP(out_channel, 32), DIV_UP(cur_nnz, 32), 1), dim3(8, 32, 1)>>>(
                    cur_nnz, in_channel, out_channel, 
                    in_feat.data_ptr<float>(), 
                    (kernel.data_ptr<float>() + k * in_channel * out_channel), 
                    out_feat.data_ptr<float>(), 
                    &in_map_ptr[cur_idx], &out_map_ptr[cur_idx]
                );
        }
        else{
          fetch_on_demand_gemm_no_fusion_fp32<32, 4, 8>
              <<<dim3(DIV_UP(out_channel, 32), DIV_UP(cur_nnz, 32), 1), dim3(8, 32, 1)>>>(
                    cur_nnz, in_channel, out_channel, 
                    in_feat.data_ptr<float>(), 
                    (kernel.data_ptr<float>() + k * in_channel * out_channel), 
                    out_feat.data_ptr<float>(), 
                    &in_map_ptr[cur_idx], &out_map_ptr[cur_idx]
                );
        }
      }
      else{
        fetch_on_demand_gemm_no_fusion_fp32_1<16, 4, 8>
              <<<dim3(DIV_UP(out_channel, 16), DIV_UP(cur_nnz, 16), 1), dim3(16, 16, 1)>>>(
                    cur_nnz, in_channel, out_channel, 
                    in_feat.data_ptr<float>(), 
                    (kernel.data_ptr<float>() + k * in_channel * out_channel), 
                    out_feat.data_ptr<float>(), 
                    &in_map_ptr[cur_idx], &out_map_ptr[cur_idx]
                );
      }
    }

    cur_idx += cur_nnz;
  }
  // precomputation only for odd channel size
  // bool precompute_mid = (input_size == output_size && k_vol % 2 == 1);
  if (precompute_mid){
    at::addmm_out(out_feat, out_feat, in_feat, kernel[mid_kernel]);
  }

  return out_feat;
}


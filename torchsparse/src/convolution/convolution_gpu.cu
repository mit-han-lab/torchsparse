#ifndef GPU_CONVOLUTION
#define GPU_CONVOLUTION
#include "../common/gpu.cuh"
#include <iostream>
#include <chrono>
#include <cstdio>

// Given each output, get an input feature for each corresponding kernel weight
// and add the output in place
__global__ void inplace_convolution(const int n, const float *in_feat,
                                    const int in_nchannel, float *out_feat,
                                    const int out_nchannel, const float *kernel,
                                    const int *neighbor_map) {
  // n = out_nchannel * out_nrows
  // The kernel computes one output scalar for each output index and each output
  // channel.
  CUDA_KERNEL_LOOP(index, n) {
    const int out_ch = index % out_nchannel;
    const int out_row = index / out_nchannel;
    // Pytorch tensors in C-ordering with in_nchannels x out_nchannels
    float tmp = 0.0;
    const float *curr_kernel = kernel + out_ch;
    const float *curr_in_feat = in_feat + out_row * in_nchannel;
    for (int in_ch = 0; in_ch < in_nchannel; in_ch++) {
      tmp += (*curr_kernel) * (*curr_in_feat);
      curr_kernel += out_nchannel;
      curr_in_feat += 1;
    }
    // Done independently, no need for atomicAdd
    out_feat[neighbor_map[out_row] * out_nchannel + out_ch] += tmp;
  }
}

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
__global__ void matmul(const float *A, const int wA, const int hA,
                       const float *B, const int wB, const int hB, float *C,
                       const int *neighbor_map, const int nmap_size,
                       const bool transpose) {
  // Use in_feat as A and kernel as B

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // Coordinate. x is for rows, y is for columns.
  const int x = BLOCK_SIZE * bx + tx;
  const int y = BLOCK_SIZE * by + ty;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub = 0;
  
  // out_npoints is  the output size.
  // conv: out_npoints <= hA; deconv: out_npoints >= hA.
  // be careful about in_row_!
  const int out_row_ = y < nmap_size ?  neighbor_map[2 * y + 1]: -1;
  const int in_row_ = y < nmap_size ? neighbor_map[2 * y] : -1;
  const int out_row = transpose ? in_row_ : out_row_;
  const int in_row = transpose ? out_row_ : in_row_;
    
  
  
  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < wA; s += BLOCK_SIZE) {
    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix
    

    As[ty][tx] = ((s + tx) < wA && y < hA && in_row >= 0) ? A[wA * in_row + s + tx] : 0;
    Bs[ty][tx] = ((s + ty) < hB && x < wB) ? B[wB * (s + ty) + x] : 0;

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    if(in_row >= 0 && out_row >= 0){
        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
    #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
          Csub += As[ty][k] * Bs[k][tx];
        }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
    
  if (out_row >= 0 && y < hA && x < wB){
    C[wB * out_row + x] += Csub;
  }
   // TODO: atomicAdd(&C[wB * out_row + x], Csub); // For conv transpose, it
  // might fail due to overlapping outputs
}

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B^T, E = D^T * A
 * wA is A's width and wB is B's width
 *
 *                +---+
 *                |B^T|
 *            +-------+
 *            |   |   |
 *            | A | C |
 *            |   |   |
 *            |   |   |
 * +------------------+
 * |    D^T   | E |
 * +----------+---+
 *
 */
__global__ void matmul2(const float *A, const int wA, const int hA,
                        const float *B, const int wB, const int hB,
                        const float *D, const int wD, const int hD, float *C,
                        float *E, const int *neighbor_map, const int nmap_size,
                        const bool transpose) {
  // Use grad_out_feat as A, transposed kernel weight as B, and in_feat as D

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // Coordinate. x is for rows, y is for columns.
  const int x = BLOCK_SIZE * bx + tx;
  const int y = BLOCK_SIZE * by + ty;

  
  const int out_row_ = y < nmap_size ?  neighbor_map[2 * y + 1]: -1;
  const int in_row_ = y < nmap_size ? neighbor_map[2 * y] : -1;
  const int out_row = transpose ? in_row_ : out_row_;
  const int in_row = transpose ? out_row_ : in_row_;
  
  
  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub = 0;
  float Esub = 0;

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ float BTs[BLOCK_SIZE][BLOCK_SIZE];

  // Declaration of the shared memory array Ds used to
  // store the sub-matrix of D
  __shared__ float DTs[BLOCK_SIZE][BLOCK_SIZE];

  // For Ds = D^T[...:..., ...:...], use the transposed grid dimension for A
  DTs[ty][tx] = (x < wD && y < hD && in_row >= 0) ? D[wD * in_row + x] : 0;

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < wA; s += BLOCK_SIZE) {
    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix
    As[ty][tx] = ((s + tx) < wA && y < hA && out_row >= 0) ? A[wA * out_row + s + tx] : 0;

    // Transposed kernel
    BTs[ty][tx] = ((s + ty) < wB && x < hB) ? B[wB * x + s + ty] : 0;

    // Synchronize to make sure the matrices are loaded
    __syncthreads();
    

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub += As[ty][k] * BTs[k][tx];
    }
    
    Esub = 0;
    
    // For Esub, reset to 0
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Esub += DTs[k][ty] * As[k][tx];
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();

    // For the E matrix which requires accmulation of multiple blocks, use
    // atomic addition. This can be replaced with a more sophisticaed reduction
    // algorithm.
    if ((bx * BLOCK_SIZE + ty) < wD && (s + tx) < wA)
      atomicAdd(&E[wA * (bx * BLOCK_SIZE + ty) + (s + tx)], Esub);
  }
    

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  if (y < hA && x < hB && in_row >= 0)
    atomicAdd(&C[hB * in_row + x], Csub);
}

void ConvolutionForwardKernelGPU(
    const float *d_in_feat, int in_nchannel, float *d_out_feat,
    int out_nchannel, const float *d_kernel,
    const int* neighbor_map,
    const int* neighbor_offset,
    const int in_npoints,
    const int out_npoints,
    const int n_neighbors,
    const bool transpose,
    cublasHandle_t cuhandle, cudaStream_t stream) {
  // For the in out buffer, use the pre allocated GPU memory space as thrust
  // resize gives segfault. Also initializing it with torch allows us to
  // allocate memory faster and efficiently.
  
  
  int kernel_volume=n_neighbors, n_active_in_volume, num_kernels, 
    neighbor_step=min(out_npoints, in_npoints);
  int cur_offset = 0;
  
  //printf("%d %d\n", in_buffer_size, in_npoints);
  
  // Iterate through each spatial kernel and get indices for in_map and out_map
  
  for (int k = 0; k < kernel_volume; k++) {
    
    n_active_in_volume = in_npoints;
    if (n_active_in_volume / SHARED_BLOCK_SIZE < 65536) {
      dim3 threads(SHARED_BLOCK_SIZE, SHARED_BLOCK_SIZE);
      dim3 grid((out_nchannel + threads.x - 1) / threads.x,
                (n_active_in_volume + threads.y - 1) / threads.y);
      matmul<<<grid, threads, 0, stream>>>(
          d_in_feat, in_nchannel, n_active_in_volume,
          &d_kernel[k * in_nchannel * out_nchannel], out_nchannel, in_nchannel,
          d_out_feat, &neighbor_map[cur_offset], neighbor_offset[k], transpose);
    } else {
      printf("call2\n");
      num_kernels = out_nchannel * n_active_in_volume;
      inplace_convolution
          <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
              num_kernels, d_in_feat, in_nchannel, d_out_feat, out_nchannel,
              &d_kernel[k * in_nchannel * out_nchannel], neighbor_map + cur_offset);
    }
    cur_offset += 2 * neighbor_offset[k];
    
  }
  
}

void ConvolutionBackwardKernelGPU(
    const float *d_in_feat, float *d_grad_in_feat, int in_nchannel,
    const float *d_grad_out_feat, int out_nchannel, float *d_kernel,
    float *d_grad_kernel, const int * neighbor_map,
    const int * neighbor_offset,
    const int in_npoints,
    const int out_npoints,
    const int n_neighbors,
    const bool transpose,
    cublasHandle_t cuhandle, cudaStream_t stream) {
  int kernel_volume=n_neighbors, n_active_in_volume;
  int neighbor_step=min(in_npoints, out_npoints);
  int cur_offset = 0;
  // Assume that old kernel will never be used.
  for (int k = 0; k < kernel_volume; k++) {
    // acceleration by setting good n_active_in_volume.
    n_active_in_volume = neighbor_offset[k];
    if (n_active_in_volume == 0)
      continue;

    
    dim3 threads(SHARED_BLOCK_SIZE, SHARED_BLOCK_SIZE);
    dim3 grid((in_nchannel + threads.x - 1) / threads.x,
              (n_active_in_volume + threads.y - 1) / threads.y);

    matmul2<<<grid, threads, 0, stream>>>(
        d_grad_out_feat, out_nchannel, n_active_in_volume, // A
        &d_kernel[k * in_nchannel * out_nchannel], out_nchannel,
        in_nchannel,                                    // B
        d_in_feat, in_nchannel, n_active_in_volume,     // D
        d_grad_in_feat,                                 // C
        &d_grad_kernel[k * in_nchannel * out_nchannel], // E
        neighbor_map + cur_offset, neighbor_offset[k], transpose);
    
    cur_offset += 2 * neighbor_offset[k];
  }

   
}

#endif

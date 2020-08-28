#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_vector.h>
#include "../common/gpu.cuh"

//input features (n, c), indices (N, 8), weight (N, 8) -> output features (N, c)
__global__ void deterministic_devoxelize_kernel(int N, int c, const int *__restrict__ indices, const float *__restrict__ weight, const float *__restrict__ feat, float *__restrict__ out){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int i = index / c;
    int j = index % c;
    
    if(i < N){
        const int* indices_ = indices + 8 * i;
        const float *weight_ = weight + 8 * i;
        const float *feat_ = feat + j;
        
        float cur_feat;
        for(int k = 0; k < 8; k++){
            cur_feat = (indices_[k] >= 0) ? feat_[indices_[k] * c]  : 0; 
            out[i * c + j] += weight_[k] * cur_feat;
        }
            
    }
    
}

//input weight (N, 8), indices (N, 8), top_grad (N, c) -> bottom grad (n, c)
__global__ void deterministic_devoxelize_grad_kernel(int N, int n, int c, const int *__restrict__ indices, const float *__restrict__ weight, const float *__restrict__ top_grad, int *__restrict__ bottom_grad){
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int i = index / c;
    int j = index % c;
    
    
    if(i < N){
        const int* indices_ = indices + 8 * i;
        const float *weight_ = weight + 8 * i;
        
        float cur_top_grad = top_grad[i * c + j];
        
        
        #pragma unroll
        for(int k = 0; k < 8; k++){
            float grad_float = weight_[k]*cur_top_grad;
            int64_t grad_int = (int64_t)round(grad_float * 1e10);
            if(indices_[k] >= 0) atomicAdd(&bottom_grad[indices_[k]*c+j], (int)grad_int);
        }
        
    
    }
}



void deterministic_devoxelize_wrapper(int N, int c, const int * indices, const float * weight, const float * feat, float * out){
    deterministic_devoxelize_kernel<<<N, c>>>(N, c, indices, weight, feat, out);
}

void deterministic_devoxelize_grad_wrapper(int N, int n, int c, const int *indices, const float * weight, const float * top_grad, int * bottom_grad){
    deterministic_devoxelize_grad_kernel<<<N, c>>>(N, n, c, indices, weight, top_grad, bottom_grad);
}

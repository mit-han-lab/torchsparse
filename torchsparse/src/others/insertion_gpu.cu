#include <stdio.h>
#include <stdlib.h>
#include <cmath>

//hashing
//input N*F float tensor, pointer to output N'*F int64 tensor, N*1 count tensor, N*1 index tensor
__global__ void insertion_kernel(int N, int c, int s, const float *__restrict__ data, const int *__restrict__ idx, const int *__restrict__ counts, float *__restrict__ out){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int i = index / c;
    int j = index % c;
    if(i < N){
        int pos = idx[i];
        if(pos < 0 || pos >= s || counts[pos] == 0) return;
        atomicAdd(&out[pos*c+j], data[i*c+j] / float(counts[pos]));
    }
}

__global__ void insertion_grad_kernel(int N, int c, int s, const float *__restrict__ top_grad, const int *__restrict__ idx, const int *__restrict__ counts, float *__restrict__ bottom_grad){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int i = index / c;
    int j = index % c;
    if(i < N){
        int pos = idx[i];
        if(pos < 0 || pos >= s || counts[pos]==0) return;
        atomicAdd(&bottom_grad[i*c+j], top_grad[pos*c+j] / float(counts[pos]));
    }
}

void insertion_wrapper(int N, int c, int s, const float * data, const int * idx, const int * counts,  float * out){
    insertion_kernel<<<N,c>>>(N, c, s, data, idx, counts, out);
}


void insertion_grad_wrapper(int N, int c, int s, const float * top_grad, const int * idx, const int * counts, float * bottom_grad){
    insertion_grad_kernel<<<N,c>>>(N,c,s,top_grad,idx,counts,bottom_grad);
}
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

//hashing
//input N*4 int32 tensor output N*1 int64 tensor
__global__ void hash_kernel(int N, const int *__restrict__ data, long int *__restrict__ out){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N){
        data += i * 4;
        unsigned long long hash = 14695981039346656037UL;
        for(int j = 0; j < 4; j++){
            hash ^= (unsigned int)data[j];
            hash *= 1099511628211UL;
        }
        hash = (hash >> 60) ^ (hash & 0xFFFFFFFFFFFFFFF);
        out[i] = hash;
    }
}


//kernel hashing: given data D and offset map K, generate D x K
//input N*4 int32 tensor, |K|*3 int32 tensor, output |K|*N int64 tensor
__global__ void kernel_hash_kernel(int N, int K, const int *__restrict__ data, const int * __restrict__ kernel_offset, long int *__restrict__ out){
    
    extern __shared__ int kernel_offset_local[];
    
    for(int i = 0; i < K * 3; i++){
        kernel_offset_local[i] = kernel_offset[i];
    }
    __syncthreads();
    
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int k = idx % K;
    int i = idx / K;
    int cur_coord[4];
    if(i < N){
        data += i * 4;
        for(int j = 0; j < 3; j++){
            cur_coord[j] = data[j]+kernel_offset[k*3+j];
        }
        cur_coord[3] = data[3];
        unsigned long long hash = 14695981039346656037UL;
        for(int j = 0; j < 4; j++){
            hash ^= (unsigned int)cur_coord[j];
            hash *= 1099511628211UL;
        }
        hash = (hash >> 60) ^ (hash & 0xFFFFFFFFFFFFFFF);
        out[k*N+i] = hash;
    }
}


void kernel_hash_wrapper(int N, int K, const int * data, const int *kernel_offset, long int * out){
    kernel_hash_kernel<<<ceil((double)(N*K)/512), 512, K*3*sizeof(int)>>>(N, K, data, kernel_offset, out);
}


void hash_wrapper(int N, const int * data, long int * out){
    hash_kernel<<<ceil((double)N/512), 512>>>(N, data, out);
}

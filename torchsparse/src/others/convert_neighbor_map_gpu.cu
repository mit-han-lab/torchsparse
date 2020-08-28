#include <stdio.h>
#include <stdlib.h>
#include <cmath>

//hashing
//input N*F float tensor, pointer to output N'*F int64 tensor, N*1 count tensor, N*1 index tensor
__global__ void convert_map_kernel(int k, int N, int N_nonzero, const int *__restrict__ nmap, const int *__restrict__ idx_batch, const int *__restrict__ idx_point, int *__restrict__ out){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(index < N_nonzero){
        int i = idx_batch[index];
        int j = idx_point[index];
        out[index << 1] = nmap[i * N + j];
        out[(index << 1) + 1] = j;
    }
}



void convert_map_wrapper(int k, int N, int N_nonzero, const int * nmap, const int * idx_batch, const int * idx_point, int * out){
    convert_map_kernel<<<int(ceil(N_nonzero / 512))+1, 512>>>(k, N, N_nonzero, nmap, idx_batch, idx_point, out);
}

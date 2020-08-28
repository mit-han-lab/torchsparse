#include <torch/torch.h>
#include <vector>
#include "hash_cpu_header.h"

void cpu_hash_wrapper(int N, const int * data, long * out){
    #pragma omp parallel for
    for(int i = 0; i < N; i++){
        unsigned long long hash = 14695981039346656037UL;
        for(int j = 0; j < 4; j++){
            hash ^= (unsigned int)data[4 * i + j];
            hash *= 1099511628211UL;
        }
        hash = (hash >> 60) ^ (hash & 0xFFFFFFFFFFFFFFF);
        out[i] = hash;
    }
}


void cpu_kernel_hash_wrapper(int N, int K, const int * data, const int *kernel_offset, long int * out){
    for(int k = 0; k < K; k++){
        #pragma omp parallel for
        for(int i = 0; i < N; i++){
            int cur_coord[4];
            for(int j = 0; j < 3; j++){
                cur_coord[j] = data[i * 4 + j]+kernel_offset[k*3+j];
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
}


at::Tensor cpu_hash_forward(
    const at::Tensor idx
)
{  
  int N = idx.size(0);
  at::Tensor out = torch::zeros({N}, at::device(idx.device()).dtype(at::ScalarType::Long));
  cpu_hash_wrapper(N, idx.data_ptr<int>(), out.data_ptr<long>());
  return out;
}


at::Tensor cpu_kernel_hash_forward(
    const at::Tensor idx,
    const at::Tensor kernel_offset
)
{  
  int N = idx.size(0);
  int K = kernel_offset.size(0);
  at::Tensor out = torch::zeros({K, N}, at::device(idx.device()).dtype(at::ScalarType::Long));
  cpu_kernel_hash_wrapper(N, K, idx.data_ptr<int>(), kernel_offset.data_ptr<int>(), out.data_ptr<long>());
  return out;
}



/*
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hash_forward", &hash_forward, "Hashing forward (CUDA)");
  m.def("kernel_hash_forward", &kernel_hash_forward, "Kernel Hashing forward (CUDA)");
}
*/


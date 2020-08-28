#ifndef _CONVERT_NEIGHBOR_MAP_CPU
#define _CONVERT_NEIGHBOR_MAP_CPU
#include <torch/torch.h>
#include <vector>


//CUDA forward declarations
void cpu_convert_map_wrapper(int k, int N, int N_nonzero, const int * nmap, const int * idx_batch, const int * idx_point, int * out);
at::Tensor cpu_convert_map_forward(
    const at::Tensor nmap,
    const at::Tensor idx_batch,
    const at::Tensor idx_point
);

#endif
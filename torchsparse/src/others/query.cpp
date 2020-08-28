#include <torch/torch.h>
#include "../hashmap/hashmap.cuh"
#include <vector>
#include <cmath>
#include <iostream>
#include "query_gpu.h"


std::vector<at::Tensor> query_forward(
    const at::Tensor hash_query,
    const at::Tensor hash_target,
    const at::Tensor idx_target
)
{
  //return group_point_forward_gpu(points, indices);
  int n = hash_target.size(0);
  int n1 = hash_query.size(0);
  int table_size =  2 * pow(2,ceil(log2((double)n)));
  if(table_size < 512){
      table_size = 512;
  }
  int num_funcs = 3;
  CuckooHashTableCuda_Multi in_hash_table(table_size, 8 * ceil(log2((double)n)),
                                                                   num_funcs); 
  at::Tensor key_buf = torch::zeros({table_size}, at::device(hash_query.device()).dtype(at::ScalarType::Long));
  at::Tensor val_buf = torch::zeros({table_size}, at::device(hash_query.device()).dtype(at::ScalarType::Long));
  at::Tensor key = torch::zeros({num_funcs*table_size}, at::device(hash_query.device()).dtype(at::ScalarType::Long));
  at::Tensor val = torch::zeros({num_funcs*table_size}, at::device(hash_query.device()).dtype(at::ScalarType::Long));
  
  in_hash_table.insert_vals((unsigned long long int*)(hash_target.data_ptr<long>()), (unsigned long long int*)(idx_target.data_ptr<long>()), (unsigned long long int*)(key_buf.data_ptr<long>()), (unsigned long long int*)(val_buf.data_ptr<long>()),  (unsigned long long int*)(key.data_ptr<long>()), (unsigned long long int*)(val.data_ptr<long>()), n);
  
  at::Tensor out = torch::zeros({n1}, at::device(hash_query.device()).dtype(at::ScalarType::Long));
  
  in_hash_table.lookup_vals((unsigned long long int*)(hash_query.data_ptr<long>()), (unsigned long long int*)(key.data_ptr<long>()), (unsigned long long int*)(val.data_ptr<long>()), (unsigned long long int*)(out.data_ptr<long>()), n1);
  return {out, key_buf, val_buf, key};
}


/*
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("query_forward", &query_forward, "hash query forward (CUDA)");
}
*/



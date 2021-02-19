#include <torch/torch.h>
#include "../hashmap/hashmap_cpu_header.hpp"
#include <vector>
#include <cmath>
#include <iostream>
#include "query_cpu_header.h"
#include <google/dense_hash_map>

at::Tensor cpu_query_forward(
    const at::Tensor hash_query,
    const at::Tensor hash_target,
    const at::Tensor idx_target)
{
  //return group_point_forward_gpu(points, indices);
  int n = hash_target.size(0);
  int n1 = hash_query.size(0);

  google::dense_hash_map<int64_t, int64_t> hashmap;
  hashmap.set_empty_key(0);
  /*
  HashTableCPU in_hash_table;
  printf("inserting %d %d...\n", n, n1);
  in_hash_table.insert_vals(hash_target.data_ptr<long>(), idx_target.data_ptr<long>(), n);
  */
  at::Tensor out = torch::zeros({n1}, at::device(hash_query.device()).dtype(at::ScalarType::Long));
  for (int idx = 0; idx < n; idx++)
  {
    int64_t key = *(hash_target.data_ptr<long>() + idx);
    int64_t val = *(idx_target.data_ptr<long>() + idx) + 1;
    hashmap.insert(std::make_pair(key, val));
  }
#pragma omp parallel for
  for (int idx = 0; idx < n1; idx++)
  {
    int64_t key = *(hash_query.data_ptr<long>() + idx);
    google::dense_hash_map<int64_t, int64_t>::iterator iter = hashmap.find(key);
    if (iter != hashmap.end())
    {
      *(out.data_ptr<long>() + idx) = iter->second;
    }
  }

  return out;
}

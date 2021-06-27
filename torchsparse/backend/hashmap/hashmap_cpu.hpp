#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <google/dense_hash_map>
#include <vector>

class HashTableCPU {
 private:
  google::dense_hash_map<int64_t, int64_t> hashmap;

 public:
  HashTableCPU() {}

  ~HashTableCPU() {}

  void insert_vals(const int64_t* const keys, const int64_t* const vals,
                   const int n);

  void lookup_vals(const int64_t* const keys, int64_t* const results,
                   const int n);
};

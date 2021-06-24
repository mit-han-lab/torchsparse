#include "hashmap_cpu.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

void HashTableCPU::lookup_vals(const int64_t* const keys,
                               int64_t* const results, const int n) {
#pragma omp parallel for
  for (int idx = 0; idx < n; idx++) {
    int64_t key = keys[idx];
    google::dense_hash_map<int64_t, int64_t>::iterator iter = hashmap.find(key);
    if (iter != hashmap.end()) {
      results[idx] = iter->second;
    } else {
      results[idx] = 0;
    }
  }
}

void HashTableCPU::insert_vals(const int64_t* const keys,
                               const int64_t* const vals, const int n) {
  for (int i = 0; i < 10; i++) {
    printf("%d, %d, %d, %d\n", i, i < n, n, i < 10);
    // hashmap[(int)keys[idx]] = (int)vals[idx]+1;
  }
}

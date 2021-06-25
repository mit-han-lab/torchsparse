#ifndef _CUCKOO_CUDA_MULTI_HPP_
#define _CUCKOO_CUDA_MULTI_HPP_

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cuda_runtime.h"

/** Reserved value for indicating "empty". */
#define EMPTY_CELL (0)
/** Max rehashing depth, and error depth. */
#define MAX_DEPTH (100)
#define ERR_DEPTH (-1)
/** CUDA naive thread block size. */
#define BLOCK_SIZE (256)
/** CUDA multi-level thread block size = bucket size. */
#define BUCKET_SIZE (512)

typedef unsigned long long int VTYPE;

/** Struct of a hash function config. */
typedef struct {
  int rv;  // Randomized XOR value.
  int ss;  // Randomized shift filter start position.
} FuncConfig;

/** Hard code hash functions and all inline helper functions for CUDA kernels'
 * use. */
inline __device__ int do_1st_hash(const VTYPE val, const int num_buckets) {
  return val % num_buckets;
}

inline __device__ int do_2nd_hash(const VTYPE val,
                                  const FuncConfig *const hash_func_configs,
                                  const int func_idx, const int size) {
  FuncConfig fc = hash_func_configs[func_idx];
  return ((val ^ fc.rv) >> fc.ss) % size;  // XOR function as 2nd-level hashing.
}

// trying to ignore EMPTY_CELL by adding 1 at make_data.
inline __device__ VTYPE fetch_val(const VTYPE data, const int pos_width) {
  return data >> pos_width;
}

inline __device__ int fetch_func(const VTYPE data, const int pos_width) {
  return data & ((0x1 << pos_width) - 1);
}

inline __device__ VTYPE make_data(const VTYPE val, const int func,
                                  const int pos_width) {
  return (val << pos_width) ^ func;
}

class CuckooHashTableCuda_Multi {
 private:
  const int _size;
  const int _evict_bound;
  const int _num_funcs;
  const int _pos_width;
  const int _num_buckets;

  FuncConfig *_d_hash_func_configs;

  /** Cuckoo hash function set. */
  FuncConfig *_hash_func_configs;

  /** Private operations. */
  void gen_hash_funcs() {
    // Calculate bit width of value range and table size.
    int val_width = 8 * sizeof(VTYPE) - ceil(log2((double)_num_funcs));
    int bucket_width = ceil(log2((double)_num_buckets));
    int size_width = ceil(log2((double)BUCKET_SIZE));
    // Generate randomized configurations.
    for (int i = 0; i < _num_funcs; ++i) {  // At index 0 is a dummy function.
      if (val_width - bucket_width <= size_width)
        _hash_func_configs[i] = {rand(), 0};
      else {
        _hash_func_configs[i] = {
            rand(), rand() % (val_width - bucket_width - size_width + 1) +
                        bucket_width};
      }
    }
  };

  inline VTYPE fetch_val(const VTYPE data) { return data >> _pos_width; }
  inline int fetch_func(const VTYPE data) {
    return data & ((0x1 << _pos_width) - 1);
  }

 public:
  CuckooHashTableCuda_Multi(const int size, const int evict_bound,
                            const int num_funcs)
      : _size(size),
        _evict_bound(evict_bound),
        _num_funcs(num_funcs),
        _pos_width(ceil(log2((double)_num_funcs))),
        _num_buckets(ceil((double)_size / BUCKET_SIZE)) {
    srand(time(NULL));
    _d_hash_func_configs = NULL;
    _hash_func_configs = NULL;
    _hash_func_configs = new FuncConfig[num_funcs];

    gen_hash_funcs();

    cudaMalloc((void **)&_d_hash_func_configs, _num_funcs * sizeof(FuncConfig));
    cudaMemcpy(_d_hash_func_configs, _hash_func_configs,
               _num_funcs * sizeof(FuncConfig), cudaMemcpyHostToDevice);
  };
  ~CuckooHashTableCuda_Multi() {
    if (_hash_func_configs != NULL) delete[] _hash_func_configs;

    if (_d_hash_func_configs != NULL) cudaFree(_d_hash_func_configs);
  };

  int insert_vals(const VTYPE *const keys, const VTYPE *const vals,
                  VTYPE *d_key_buf, VTYPE *d_val_buf, VTYPE *d_key,
                  VTYPE *d_val, const int n);

  void lookup_vals(const VTYPE *const keys, VTYPE *const results, VTYPE *d_key,
                   VTYPE *d_val, const int n);
};

__global__ void cuckooBucketKernel_Multi(VTYPE *const key_buf,
                                         VTYPE *const val_buf, const int size,
                                         const VTYPE *const keys,
                                         const VTYPE *const vals, const int n,
                                         int *const counters,
                                         const int num_buckets);

__global__ void cuckooInsertKernel_Multi(
    VTYPE *const key, VTYPE *const val, const VTYPE *const key_buf,
    const VTYPE *const val_buf, const int size,
    const FuncConfig *const hash_func_configs, const int num_funcs,
    const int *const counters, const int num_buckets, const int evict_bound,
    const int pos_width, int *const rehash_requests);

__global__ void cuckooLookupKernel_Multi(
    const VTYPE *const keys, VTYPE *const results, const int n,
    const VTYPE *const all_keys, const VTYPE *const all_vals, const int size,
    const FuncConfig *const hash_func_configs, const int num_funcs,
    const int num_buckets, const int pos_width);

#endif

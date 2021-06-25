#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "../utils/atomic.cuh"
#include "hashmap_cuda.cuh"

__global__ void cuckooBucketKernel_Multi(
    uint64_t *const key_buf, uint64_t *const val_buf, const int size,
    const uint64_t *const keys, const uint64_t *const vals, const int n,
    int *const counters, const int num_buckets) {
  // Get thread index.
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Only threads within range are active.
  if (idx < n) {
    // Do 1st-level hashing to get bucket id, then do atomic add to get index
    // inside the bucket.
    uint64_t key = keys[idx];
    uint64_t val = vals[idx];

    int bucket_num = do_1st_hash(key, num_buckets);
    int bucket_ofs = atomicAdd(&counters[bucket_num], 1);

    // Directly write the key into the table buffer.
    if (bucket_ofs >= BUCKET_SIZE) {
      printf("%d/%d ERROR: bucket overflow! (n=%d, bucket_num=%d/%d, key=%d)\n",
             bucket_ofs, BUCKET_SIZE, n, bucket_num, num_buckets, key);
    } else {
      key_buf[bucket_num * BUCKET_SIZE + bucket_ofs] = key;
      val_buf[bucket_num * BUCKET_SIZE + bucket_ofs] = val;
    }
  }
}

__global__ void cuckooInsertKernel_Multi(
    uint64_t *const key, uint64_t *const val, const uint64_t *const key_buf,
    const uint64_t *const val_buf, const int size,
    const FuncConfig *const hash_func_configs, const int num_funcs,
    const int *const counters, const int num_buckets, const int evict_bound,
    const int pos_width, int *const rehash_requests) {
  // Create local cuckoo table in shared memory. Size passed in as the third
  // kernel parameter.
  extern __shared__ uint64_t local_key[];
  for (int i = 0; i < num_funcs; ++i) {
    local_key[i * BUCKET_SIZE + threadIdx.x] = EMPTY_CELL;
  }

  // might be useful
  __syncthreads();

  // Get thread index.
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  uint64_t cur_idx = idx;

  // Only threads within local bucket range are active.
  if (threadIdx.x < counters[blockIdx.x]) {
    // Set initial conditions.
    uint64_t cur_key = key_buf[cur_idx];
    int cur_func = 0;
    int evict_count = 0;

    // Start the test-kick-and-reinsert loops.
    do {
      int pos = do_2nd_hash(cur_key, hash_func_configs, cur_func, BUCKET_SIZE);

      uint64_t new_data = make_data(cur_idx + 1, cur_func, pos_width);

      uint64_t old_idx =
          atomicExch(&local_key[cur_func * BUCKET_SIZE + pos], new_data);

      if (old_idx != EMPTY_CELL) {
        cur_idx = fetch_val(old_idx, pos_width) - 1;
        // potential overflow here. It seems that cur_idx < 0 is possible!
        cur_key = key_buf[cur_idx];
        cur_func = (fetch_func(old_idx, pos_width) + 1) % num_funcs;
        evict_count++;
      } else {
        break;
      }

    } while (evict_count < num_funcs * evict_bound);

    // If exceeds eviction bound, then needs rehashing.
    if (evict_count >= num_funcs * evict_bound) {
      atomicAdd(rehash_requests, 1);
    }
  }

  // Every thread write its responsible local slot into the global data table.
  __syncthreads();
  for (int i = 0; i < num_funcs; ++i) {
    uint64_t cur_idx = local_key[i * BUCKET_SIZE + threadIdx.x];
    if (cur_idx == EMPTY_CELL) {
      continue;
    }
    int cur_func = fetch_func(cur_idx, pos_width);
    cur_idx = fetch_val(cur_idx, pos_width) - 1;
    key[i * size + idx] = key_buf[cur_idx];
    val[i * size + idx] = val_buf[cur_idx];
  }
}

__global__ void cuckooLookupKernel_Multi(
    const uint64_t *const keys, uint64_t *const results, const int n,
    const uint64_t *const all_keys, const uint64_t *const all_vals,
    const int size, const FuncConfig *const hash_func_configs,
    const int num_funcs, const int num_buckets, const int pos_width) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Only threads within range are active.
  if (idx < n) {
    uint64_t key = keys[idx];
    int bucket_num = do_1st_hash(key, num_buckets);
    for (int i = 0; i < num_funcs; ++i) {
      int pos = bucket_num * BUCKET_SIZE +
                do_2nd_hash(key, hash_func_configs, i, BUCKET_SIZE);
      if (all_keys[i * size + pos] == key) {
        results[idx] = all_vals[i * size + pos] + 1;
        return;
      }
    }

    // TODO(Haotian): should be a value that will not be encountered.
    results[idx] = EMPTY_CELL;
  }
}

void CuckooHashTableCuda_Multi::lookup_vals(const uint64_t *const keys,
                                            uint64_t *d_key, uint64_t *d_val,
                                            uint64_t *const results,
                                            const int n) {
  // Launch the lookup kernel.
  cuckooLookupKernel_Multi<<<ceil((double)n / BUCKET_SIZE), BUCKET_SIZE>>>(
      keys, results, n, d_key, d_val, _size, _d_hash_func_configs, _num_funcs,
      _num_buckets, _pos_width);
}

int CuckooHashTableCuda_Multi::insert_vals(const uint64_t *const keys,
                                           const uint64_t *const vals,
                                           uint64_t *d_key_buf,
                                           uint64_t *d_val_buf, uint64_t *d_key,
                                           uint64_t *d_val, const int n) {
  //
  // Phase 1: Distribute keys into buckets.
  //

  // Allocate GPU memory.

  int *d_counters = NULL;

  cudaMalloc((void **)&d_counters, _num_buckets * sizeof(int));

  cudaMemset(d_counters, 0, _num_buckets * sizeof(int));

  // Invoke bucket kernel.
  cuckooBucketKernel_Multi<<<ceil((double)n / BUCKET_SIZE), BUCKET_SIZE>>>(
      d_key_buf, d_val_buf, _size, keys, vals, n, d_counters, _num_buckets);

  //
  // Phase 2: Local cuckoo hashing.
  //

  // Allocate GPU memory.

  cudaDeviceSynchronize();
  int *d_rehash_requests = NULL;

  cudaMalloc((void **)&d_rehash_requests, sizeof(int));

  // Copy values onto GPU memory.
  cudaMemcpy(_d_hash_func_configs, _hash_func_configs,
             _num_funcs * sizeof(FuncConfig), cudaMemcpyHostToDevice);

  // Invoke insert kernel. Passes shared memory table size by the third
  // argument. Loops until no rehashing needed.

  int rehash_count = 0;
  do {
    int rehash_requests = 0;
    cudaMemset(d_rehash_requests, 0, sizeof(int));
    cuckooInsertKernel_Multi<<<ceil((double)_size / BUCKET_SIZE), BUCKET_SIZE,
                               _num_funcs * BUCKET_SIZE * sizeof(uint64_t)>>>(
        d_key, d_val, d_key_buf, d_val_buf, _size, _d_hash_func_configs,
        _num_funcs, d_counters, _num_buckets, _evict_bound, _pos_width,
        d_rehash_requests);
    cudaMemcpy(&rehash_requests, d_rehash_requests, sizeof(int),
               cudaMemcpyDeviceToHost);

    if (rehash_requests == 0) {
      break;
    } else {
      rehash_count++;
      gen_hash_funcs();
      cudaMemcpy(_d_hash_func_configs, _hash_func_configs,
                 _num_funcs * sizeof(FuncConfig), cudaMemcpyHostToDevice);
    }
  } while (rehash_count < MAX_DEPTH);

  cudaDeviceSynchronize();

  // Free GPU resources.

  if (d_counters != NULL) {
    cudaFree(d_counters);
  }
  if (d_rehash_requests != NULL) {
    cudaFree(d_rehash_requests);
  }

  return (rehash_count < MAX_DEPTH) ? rehash_count : ERR_DEPTH;
}

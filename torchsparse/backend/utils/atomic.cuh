#pragma once

__device__ static uint64_t atomicExch(uint64_t *addr, uint64_t val) {
  return (uint64_t)atomicExch((unsigned long long int *)addr,
                              (unsigned long long int)val);
}

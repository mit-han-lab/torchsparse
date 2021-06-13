#include <cuda_runtime.h>
#include <chrono>
#include <thread>
#include <stdio.h>
#include "tools.h"


std::chrono::high_resolution_clock::time_point get_time(){
    cudaDeviceSynchronize();
    auto ts = std::chrono::high_resolution_clock::now();
    return ts;
}
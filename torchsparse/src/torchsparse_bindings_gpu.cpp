#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "convolution/convolution_cpu_header.h"
#include "hash/hash_cpu_header.h"
#include "others/convert_neighbor_map_cpu_header.h"
#include "others/insertion_cpu_header.h"
#include "others/query_cpu_header.h"
#include "convolution/convolution_gpu.h"
#include "hash/hash_gpu.h"
#include "interpolation/devox_gpu.h"
#include "interpolation/devox_cpu_header.h"
#include "others/convert_neighbor_map_gpu.h"
#include "others/count_gpu.h"
#include "others/insertion_gpu.h"
#include "others/insertion_cpu_header.h"
#include "others/query_gpu.h"
#include "others/count_cpu_header.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("sparseconv_cpu_forward", &ConvolutionForwardCPU, "point cloud convolution forward (CPU)");
    m.def("sparseconv_cpu_backward", &ConvolutionBackwardCPU, "point cloud convolution backward (CPU)");
    m.def("cpu_kernel_hash_forward", &cpu_kernel_hash_forward, "Kernel Hashing forward (CPU)");
    m.def("cpu_convert_map_forward", &cpu_convert_map_forward, "Convert neighbor map forward (CPU)");
    m.def("cpu_insertion_forward", &cpu_insertion_forward, "Insertion forward (CPU)");
    m.def("cpu_insertion_backward", &cpu_insertion_backward, "Insertion backward (CPU)");
    m.def("cpu_query_forward", &cpu_query_forward, "hash query forward (CPU)");
    m.def("sparseconv_forward", &ConvolutionForwardGPU, "point cloud convolution forward (CUDA)");
    m.def("sparseconv_backward", &ConvolutionBackwardGPU, "point cloud convolution backward (CUDA)");
    m.def("hash_forward", &hash_forward, "Hashing forward (CUDA)");
    m.def("kernel_hash_forward", &kernel_hash_forward, "Kernel Hashing forward (CUDA)");
    m.def("cpu_hash_forward", &cpu_hash_forward, "Hashing forward (CPU)");
    m.def("devoxelize_forward", &devoxelize_forward, "Devoxelization forward (CUDA)");
    m.def("devoxelize_backward", &devoxelize_backward, "Devoxelization backward (CUDA)");
    m.def("deterministic_devoxelize_forward", &deterministic_devoxelize_forward, "Devoxelization forward (CUDA)");
    m.def("deterministic_devoxelize_backward", &deterministic_devoxelize_backward, "Devoxelization backward (CUDA)");
    m.def("cpu_devoxelize_forward", &cpu_devoxelize_forward, "Devoxelization forward (CPU)");
    m.def("cpu_devoxelize_backward", &cpu_devoxelize_backward, "Devoxelization backward (CPU)");
    m.def("count_forward", &count_forward, "Counting forward (CUDA)");
    m.def("cpu_count_forward", &cpu_count_forward, "count forward (CPU)");
    m.def("insertion_forward", &insertion_forward, "Insertion forward (CUDA)");
    m.def("insertion_backward", &insertion_backward, "Insertion backward (CUDA)");
    m.def("cpu_insertion_forward", &cpu_insertion_forward, "Insertion forward (CPU)");
    m.def("cpu_insertion_backward", &cpu_insertion_backward, "Insertion backward (CPU)");
    m.def("query_forward", &query_forward, "hash query forward (CUDA)");
    m.def("convert_map_forward", &convert_map_forward, "Convert neighbor map forward (CUDA)");
}

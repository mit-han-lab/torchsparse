#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "convolution/convolution_cpu_header.h"
#include "hash/hash_cpu_header.h"
#include "interpolation/devox_cpu_header.h"
#include "others/insertion_cpu_header.h"
#include "others/query_cpu_header.h"
#include "others/count_cpu_header.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparseconv_cpu_forward", &ConvolutionForwardCPU, "point cloud convolution forward (CPU)");
    m.def("sparseconv_cpu_backward", &ConvolutionBackwardCPU, "point cloud convolution backward (CPU)");
    m.def("cpu_hash_forward", &cpu_hash_forward, "Hashing forward (CPU)");
    m.def("cpu_kernel_hash_forward", &cpu_kernel_hash_forward, "Kernel Hashing forward (CPU)");
    m.def("cpu_insertion_forward", &cpu_insertion_forward, "Insertion forward (CPU)");
    m.def("cpu_insertion_backward", &cpu_insertion_backward, "Insertion backward (CPU)");
    m.def("cpu_devoxelize_forward", &cpu_devoxelize_forward, "Devoxelization forward (CPU)");
    m.def("cpu_devoxelize_backward", &cpu_devoxelize_backward, "Devoxelization backward (CPU)");
    m.def("cpu_query_forward", &cpu_query_forward, "hash query forward (CPU)");
    m.def("cpu_count_forward", &cpu_count_forward, "count forward (CPU)");
}



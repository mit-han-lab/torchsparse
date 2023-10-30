#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "convolution/convolution_gather_scatter_cpu.h"
#include "devoxelize/devoxelize_cpu.h"
#include "hash/hash_cpu.h"
#include "others/count_cpu.h"
#include "others/query_cpu.h"
#include "voxelize/voxelize_cpu.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv_forward_gather_scatter_cpu", &conv_forward_gather_scatter_cpu);
  m.def("conv_backward_gather_scatter_cpu", &conv_backward_gather_scatter_cpu);
  m.def("voxelize_forward_cpu", &voxelize_forward_cpu);
  m.def("voxelize_backward_cpu", &voxelize_backward_cpu);
  m.def("devoxelize_forward_cpu", &devoxelize_forward_cpu);
  m.def("devoxelize_backward_cpu", &devoxelize_backward_cpu);
  m.def("hash_cpu", &hash_cpu);
  m.def("kernel_hash_cpu", &kernel_hash_cpu);
  m.def("hash_query_cpu", &hash_query_cpu);
  m.def("count_cpu", &count_cpu);
}

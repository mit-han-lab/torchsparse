#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "convolution/convolution_cpu.h"
#include "convolution/convolution_cuda.h"
#include "devoxelize/devoxelize_cpu.h"
#include "devoxelize/devoxelize_cuda.h"
#include "hash/hash_cpu.h"
#include "hash/hash_cuda.h"
#include "others/count_cpu.h"
#include "others/count_cuda.h"
#include "others/query_cpu.h"
#include "others/query_cuda.h"
#include "voxelize/voxelize_cpu.h"
#include "voxelize/voxelize_cuda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("convolution_forward_cpu", &convolution_forward_cpu);
  m.def("convolution_forward_cuda", &convolution_forward_cuda);
  m.def("convolution_backward_cpu", &convolution_backward_cpu);
  m.def("convolution_backward_cuda", &convolution_backward_cuda);
  m.def("voxelize_forward_cpu", &voxelize_forward_cpu);
  m.def("voxelize_forward_cuda", &voxelize_forward_cuda);
  m.def("voxelize_backward_cpu", &voxelize_backward_cpu);
  m.def("voxelize_backward_cuda", &voxelize_backward_cuda);
  m.def("devoxelize_forward_cpu", &devoxelize_forward_cpu);
  m.def("devoxelize_forward_cuda", &devoxelize_forward_cuda);
  m.def("devoxelize_backward_cpu", &devoxelize_backward_cpu);
  m.def("devoxelize_backward_cuda", &devoxelize_backward_cuda);
  m.def("hash_cpu", &hash_cpu);
  m.def("hash_cuda", &hash_cuda);
  m.def("kernel_hash_cpu", &kernel_hash_cpu);
  m.def("kernel_hash_cuda", &kernel_hash_cuda);
  m.def("hash_query_cpu", &hash_query_cpu);
  m.def("hash_query_cuda", &hash_query_cuda);
  m.def("count_cpu", &count_cpu);
  m.def("count_cuda", &count_cuda);
}

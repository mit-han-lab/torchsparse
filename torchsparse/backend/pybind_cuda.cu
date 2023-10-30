#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "convolution/convolution_gather_scatter_cpu.h"
#include "convolution/convolution_gather_scatter_cuda.h"
#include "convolution/convolution_forward_fetch_on_demand_cuda.h"
#include "convolution/convolution_forward_implicit_gemm_cuda.h"
#include "convolution/convolution_forward_implicit_gemm_sorted_cuda.h"
#include "convolution/convolution_backward_wgrad_implicit_gemm_cuda.h"
#include "convolution/convolution_backward_wgrad_implicit_gemm_sorted_cuda.h"
#include "devoxelize/devoxelize_cpu.h"
#include "devoxelize/devoxelize_cuda.h"
#include "hash/hash_cpu.h"
#include "hash/hash_cuda.h"
#include "others/count_cpu.h"
#include "others/count_cuda.h"
#include "others/downsample_cuda.h"
#include "others/exclusive_scan_cuda.h"
#include "others/query_cpu.h"
#include "others/query_cuda.h"
#include "others/reduce_bitmask_cuda.h"
#include "others/reorder_map_cuda.h"
#include "others/sparsemapping_cuda.h"
#include "voxelize/voxelize_cpu.h"
#include "voxelize/voxelize_cuda.h"
#include "hashmap/hashmap_cuda.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<hashtable>(m, "GPUHashTable")
        .def(py::init<const int>())
        .def(py::init<torch::Tensor, torch::Tensor>())
        .def("insert_vals", &hashtable::insert_vals)
        .def("lookup_vals", &hashtable::lookup_vals)
        .def("insert_coords", &hashtable::insert_coords)
        .def("lookup_coords", &hashtable::lookup_coords);
  py::class_<hashtable32>(m, "GPUHashTable32")
        .def(py::init<const int>())
        .def(py::init<torch::Tensor, torch::Tensor>())
        .def("insert_vals", &hashtable32::insert_vals)
        .def("lookup_vals", &hashtable32::lookup_vals)
        .def("insert_coords", &hashtable32::insert_coords)
        .def("lookup_coords", &hashtable32::lookup_coords);
  m.def("conv_forward_gather_scatter_cpu", &conv_forward_gather_scatter_cpu);
  m.def("conv_forward_gather_scatter_cuda", &conv_forward_gather_scatter_cuda);
  m.def("conv_forward_fetch_on_demand_cuda", &conv_forward_fetch_on_demand_cuda);
  m.def("conv_forward_fetch_on_demand_no_fusion_cuda", &conv_forward_fetch_on_demand_no_fusion_cuda);
  m.def("conv_forward_implicit_gemm_cuda", &conv_forward_implicit_gemm_cuda, py::arg("_in_feats"), py::arg("_kernel"), py::arg("_out_in_map"), py::arg("num_out_feats"),py::arg("num_out_channels"), py::arg("allow_tf32") = false, py::arg("allow_fp16") = true);
  m.def("conv_forward_implicit_gemm_sorted_cuda", &conv_forward_implicit_gemm_sorted_cuda, py::arg("_in_feats"), py::arg("_kernel"), py::arg("_out_in_map"), py::arg("_reduced_mask"), py::arg("_reorder_loc"), py::arg("num_out_feats"), py::arg("num_out_channels"), py::arg("allow_tf32") = false, py::arg("allow_fp16") = true);
  m.def("conv_backward_wgrad_implicit_gemm_cuda", &conv_backward_wgrad_implicit_gemm_cuda, py::arg("_in_feats"), py::arg("_kernel"), py::arg("_out_in_map"), py::arg("split_k_iters"), py::arg("allow_tf32") = false, py::arg("allow_fp16") = true);
  m.def("conv_backward_wgrad_implicit_gemm_sorted_cuda", &conv_backward_wgrad_implicit_gemm_sorted_cuda, py::arg("_in_feats"), py::arg("_kernel"), py::arg("_out_in_map"), py::arg("_reduced_mask"), py::arg("_reorder_loc"), py::arg("split_k_iters"), py::arg("allow_tf32") = false, py::arg("allow_fp16") = true);
  m.def("conv_backward_gather_scatter_cpu", &conv_backward_gather_scatter_cpu);
  m.def("conv_backward_gather_scatter_cuda", &conv_backward_gather_scatter_cuda);
  m.def("voxelize_forward_cpu", &voxelize_forward_cpu);
  m.def("voxelize_forward_cuda", &voxelize_forward_cuda);
  m.def("voxelize_backward_cpu", &voxelize_backward_cpu);
  m.def("voxelize_backward_cuda", &voxelize_backward_cuda);
  m.def("to_dense_forward_cuda", &to_dense_forward_cuda);
  m.def("to_dense_backward_cuda", &to_dense_backward_cuda);
  m.def("devoxelize_forward_cpu", &devoxelize_forward_cpu);
  m.def("devoxelize_forward_cuda", &devoxelize_forward_cuda);
  m.def("devoxelize_backward_cpu", &devoxelize_backward_cpu);
  m.def("devoxelize_backward_cuda", &devoxelize_backward_cuda);
  m.def("exclusive_scan_quantified_wrapper", &exclusive_scan_quantified_wrapper);
  m.def("hash_cpu", &hash_cpu);
  m.def("hash_cuda", &hash_cuda);
  m.def("kernel_hash_cpu", &kernel_hash_cpu);
  m.def("kernel_hash_cuda", &kernel_hash_cuda);
  m.def("hash_query_cpu", &hash_query_cpu);
  m.def("hash_query_cuda", &hash_query_cuda);
  m.def("convert_transposed_out_in_map", &convert_transposed_out_in_map);
  m.def("derive_bitmask_from_out_in_map", &derive_bitmask_from_out_in_map);
  m.def("reduce_bitmask_cuda", &reduce_bitmask_cuda);
  m.def("reorder_out_in_map_cuda", &reorder_out_in_map_cuda);
  m.def("build_kernel_map_subm_hashmap", &build_kernel_map_subm_hashmap);
  m.def("build_kernel_map_downsample_hashmap", &build_kernel_map_downsample_hashmap);
  m.def("build_kernel_map_subm_hashmap_int32", &build_kernel_map_subm_hashmap_int32);
  m.def("build_kernel_map_downsample_hashmap_int32", &build_kernel_map_downsample_hashmap_int32);
  m.def("build_mask_from_kmap", &build_mask_from_kmap);
  m.def("downsample_cuda", &downsample_cuda);
  m.def("count_cpu", &count_cpu);
  m.def("count_cuda", &count_cuda);
}

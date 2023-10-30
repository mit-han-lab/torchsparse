#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <torch/extension.h>

#include <algorithm>
#include <chrono>

#include "convolution_gather_scatter_cuda.h"

#define CONVERT_FLOAT(pointer) (reinterpret_cast<float *>(&(pointer))[0])
#define CONVERT_HALF2(pointer) (reinterpret_cast<half2 *>(&(pointer))[0])
#define CONVERT_INT4(pointer) (reinterpret_cast<int4 *>(&(pointer))[0])

template <typename scalar_t>
__global__ void gather_kernel(const int n_k, const int n_in, const int c,
                              scalar_t *in_feat, scalar_t *out_feat,
                              const int *kmap, const bool transpose) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  bool isfloat = sizeof(scalar_t) == 4;
  int i, j;
  if (isfloat) {
    i = index / c;
    j = index % c;
  } else {
    i = index / (c >> 1);
    j = index % (c >> 1);
  }
  if (i >= n_k) return;
  int in_pos = kmap[2 * i + transpose];
  if (in_pos < 0) return;
  if (isfloat) {
    out_feat[i * c + j] = in_feat[in_pos * c + j];
  } else {
    CONVERT_HALF2(out_feat[i * c + (j << 1)]) =
        CONVERT_HALF2(in_feat[in_pos * c + (j << 1)]);
  }
}

template <typename scalar_t>
__global__ void scatter_kernel(const int n_in, const int n_out, const int c,
                               scalar_t *in_feat, scalar_t *out_feat,
                               const int *kmap, const bool transpose) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int i, j;
  bool isfloat = sizeof(scalar_t) == 4;
  if (isfloat) {
    i = index / c;
    j = index % c;
  } else {
    i = index / (c >> 1);
    j = index % (c >> 1);
  }
  if (i >= n_in) return;
  int out_pos = kmap[2 * i + 1 - transpose];
  if (out_pos < 0 || out_pos >= n_out) return;
  if (isfloat) {
    out_feat[out_pos * c + j] += in_feat[i * c + j];
  } else {
    half2 cur_out_feat = CONVERT_HALF2(out_feat[out_pos * c + (j << 1)]);
    cur_out_feat =
        __hadd2(cur_out_feat, CONVERT_HALF2(in_feat[i * c + (j << 1)]));
    CONVERT_HALF2(out_feat[out_pos * c + (j << 1)]) = cur_out_feat;
  }
}

// fused gather
template <typename scalar_t>
__global__ void gather_all_kernel_pad_sep_with_mask(
    const int n, const int c, const int kernel_volume, scalar_t *in_feat,
    scalar_t *out_feat, const int *kmap, const int *kmap_sizes,
    const int *cum_kmap_sizes, const int *cum_buffer_sizes,
    const int *input_mask, const int *output_mask, const bool transpose,
    const bool precompute_mid) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int i, j;
  int offset = (sizeof(scalar_t) == 4) ? 2 : 3;
  i = index / (c >> offset);
  j = index % (c >> offset);
  if (i >= n) return;
  int4 tmps[1];
  CONVERT_INT4(tmps) = CONVERT_INT4(in_feat[i * c + (j << offset)]);
  if (transpose) {
    for (int k = 0; k < kernel_volume; k++) {
      // if(precompute_mid && k == kernel_volume / 2) continue;
      // int input_kmap_pos = input_mask[i * kernel_volume + k];
      //  another layout
      int input_kmap_pos = output_mask[k * n + i];
      if (input_kmap_pos < 0) continue;
      int cum_buffer_size = cum_buffer_sizes[k];
      // CONVERT_HALF2(out_feat[(cum_buffer_size + input_kmap_pos) * c + (j <<
      // 1)]) = CONVERT_HALF2(in_feat[i * c + (j << 1)]);
      CONVERT_INT4(
          out_feat[(cum_buffer_size + input_kmap_pos) * c + (j << offset)]) =
          tmps[0];
    }
  } else {
    for (int k = 0; k < kernel_volume; k++) {
      if (precompute_mid && k == kernel_volume / 2) continue;
      // int input_kmap_pos = input_mask[i * kernel_volume + k];
      //  another layout
      int input_kmap_pos = input_mask[k * n + i];
      if (input_kmap_pos < 0) continue;
      int cum_buffer_size = cum_buffer_sizes[k];
      CONVERT_INT4(
          out_feat[(cum_buffer_size + input_kmap_pos) * c + (j << offset)]) =
          tmps[0];
    }
  }
}

// fused scatter
template <typename scalar_t>
__global__ void scatter_all_kernel_pad_sep(
    const int c, const int kernel_volume, scalar_t *in_feat, scalar_t *out_feat,
    const int *kmap, const int *kmap_sizes, const int *cum_kmap_sizes,
    const int *cum_buffer_sizes, const bool transpose,
    const bool precompute_mid) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  bool isfloat = sizeof(scalar_t) == 4;
  int i, j;
  if (isfloat) {
    i = index / c;
    j = index % c;
  } else {
    i = index / (c >> 1);
    j = index % (c >> 1);
  }
  // #pragma unroll
  for (int k = 0; k < kernel_volume; k++) {
    if (precompute_mid && k == kernel_volume / 2) continue;
    int cur_kmap_size = kmap_sizes[k];
    int cum_kmap_size = k > 0 ? cum_kmap_sizes[k - 1] : 0;
    int cum_buffer_size = cum_buffer_sizes[k];
    if (i >= cur_kmap_size) continue;
    int out_pos = kmap[((cum_kmap_size + i) << 1) + 1 - transpose];
    if (out_pos < 0) continue;
    if (isfloat) {
      atomicAdd(&CONVERT_FLOAT(out_feat[out_pos * c + j]),
                CONVERT_FLOAT(in_feat[(cum_buffer_size + i) * c + j]));
    } else {
      atomicAdd(&CONVERT_HALF2(out_feat[out_pos * c + (j << 1)]),
                CONVERT_HALF2(in_feat[(cum_buffer_size + i) * c + (j << 1)]));
    }
  }
}

__global__ void scatter_all_kernel_pad_sep_with_mask_half(
    const int n, const int c, const int kernel_volume, half *in_feat,
    half *out_feat, const int *kmap, const int *kmap_sizes,
    const int *cum_kmap_sizes, const int *cum_buffer_sizes,
    const int *input_mask, const int *output_mask, const bool transpose,
    const bool precompute_mid) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int i, j;
  i = index / (c >> 3);
  j = index % (c >> 3);
  // half2 tmp(__float2half(0.0f), __float2half(0.0f));
  half2 tmps[4];
  int4 tmps_int4[1];
  for (int k = 0; k < 4; k++) tmps[k].x = tmps[k].y = __float2half(0.0f);
  if (i >= n) return;
  if (transpose) {
    for (int k = 0; k < kernel_volume; k++) {
      if (precompute_mid && k == kernel_volume / 2) continue;
      // int output_kmap_pos = output_mask[i * kernel_volume + k];
      // another layout
      int output_kmap_pos = input_mask[k * n + i];
      if (output_kmap_pos < 0) continue;
      int cum_buffer_size = cum_buffer_sizes[k];
      tmps_int4[0] = CONVERT_INT4(
          in_feat[(cum_buffer_size + output_kmap_pos) * c + (j << 3)]);
#pragma unroll
      for (int l = 0; l < 4; l++) {
        tmps[l] = __hadd2(tmps[l], *(reinterpret_cast<half2 *>(tmps_int4) + l));
      }
    }
  } else {
    for (int k = 0; k < kernel_volume; k++) {
      if (precompute_mid && k == kernel_volume / 2) continue;
      // int output_kmap_pos = output_mask[i * kernel_volume + k];
      // another layout
      int output_kmap_pos = output_mask[k * n + i];
      if (output_kmap_pos < 0) continue;
      int cum_buffer_size = cum_buffer_sizes[k];
      tmps_int4[0] = CONVERT_INT4(
          in_feat[(cum_buffer_size + output_kmap_pos) * c + (j << 3)]);
#pragma unroll
      for (int l = 0; l < 4; l++) {
        tmps[l] = __hadd2(tmps[l], *(reinterpret_cast<half2 *>(tmps_int4) + l));
      }
    }
  }
  CONVERT_INT4(out_feat[i * c + (j << 3)]) = CONVERT_INT4(tmps);
}

__global__ void scatter_all_kernel_pad_sep_with_mask_float(
    const int n, const int c, const int kernel_volume, float *in_feat,
    float *out_feat, const int *kmap, const int *kmap_sizes,
    const int *cum_kmap_sizes, const int *cum_buffer_sizes,
    const int *input_mask, const int *output_mask, const bool transpose,
    const bool precompute_mid) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int i, j;
  i = index / (c >> 2);
  j = index % (c >> 2);
  float tmp = 0.0f;
  if (i >= n) return;

  float tmps[4];
  int4 tmps_int4[1];
  for (int k = 0; k < 4; k++) tmps[k] = 0;
  if (i >= n) return;
  if (transpose) {
    for (int k = 0; k < kernel_volume; k++) {
      if (precompute_mid && k == kernel_volume / 2) continue;
      // int output_kmap_pos = output_mask[i * kernel_volume + k];
      //  another layout
      int output_kmap_pos = input_mask[k * n + i];
      if (output_kmap_pos < 0) continue;
      int cum_buffer_size = cum_buffer_sizes[k];
      tmps_int4[0] = CONVERT_INT4(
          in_feat[(cum_buffer_size + output_kmap_pos) * c + (j << 2)]);
#pragma unroll
      for (int l = 0; l < 4; l++) {
        tmps[l] += *(reinterpret_cast<float *>(tmps_int4) + l);
      }
    }
  } else {
    for (int k = 0; k < kernel_volume; k++) {
      if (precompute_mid && k == kernel_volume / 2) continue;
      // int output_kmap_pos = output_mask[i * kernel_volume + k];
      //  another layout
      int output_kmap_pos = output_mask[k * n + i];
      if (output_kmap_pos < 0) continue;
      int cum_buffer_size = cum_buffer_sizes[k];
      tmps_int4[0] = CONVERT_INT4(
          in_feat[(cum_buffer_size + output_kmap_pos) * c + (j << 2)]);
#pragma unroll
      for (int l = 0; l < 4; l++) {
        tmps[l] += *(reinterpret_cast<float *>(tmps_int4) + l);
      }
    }
  }
  CONVERT_INT4(out_feat[i * c + (j << 2)]) = CONVERT_INT4(tmps);
  // float verify = *(reinterpret_cast<float*>(tmps) + 3);
  // printf("%f %f\n", verify, out_feat[i * c + (j << 2) + 3]);
}

// in_feat: (N, c) N=# of input points, c = input channels
// out_feat: (M, o) M=# of output points, o = output channels
//                  for stride=1, M=N. For stride>1, the N input coords
//                  are requantized to M points with grid size (stride *
//                  cur_stride)
// kernel: (k^3, c, o) for a 3D convolution of length k
// neighbor_map: (a, 2) the hash table query results from out_coords to
//                      in_coords
//                      where neighbor_map[:,0] is the index of the output
//                      feature and neighbor_map[:,1] is the index of the input
//                      feature
// neighbor_offset: (k^3) count of active weights based on neighbor_map
//                      with unused weights having 0 and neighbor_offset[k^3/2]
//                      holding w[0,0].
// epsilon: tolerance of redundant computation in adaptive matmul grouping
//                      start a new group when the
//                      redundant computation ratio > epsilon
// mm_thresh: threshold of the maximum workload size within each group
//                      perform bmm if the workload size smaller than mm_thresh
//                      and perform mm otherwise
// conv_mode: which conv backend to use by default
//                      0: fallback to torchsparse 1.4 conv w/o optimizations
//                      1: conv with fused locality-aware gather-scatter
//                      2: conv with all latest optimizations, including
//                         fused gather-scatter and matmul grouping. Kernel
//                         reordering should be done in piror steps for
//                         grouping to function properly
at::Tensor conv_forward_gather_scatter_cuda(
    at::Tensor in_feat, at::Tensor kernel, at::Tensor neighbor_map,
    at::Tensor neighbor_offset, at::Tensor input_mask, at::Tensor output_mask,
    const int output_size, const float epsilon, const int mm_thresh,
    const int conv_mode, const bool transpose, at::Tensor global_buffer) {
  int buffer_size = (int)torch::sum(neighbor_offset).item<int>();
  // be careful about the fallback setting

  // [!!!] NOTE: be careful, current buffer_size calculation is wrong, it does
  // not take into consideration padding!
  // if(1){
  if (conv_mode == 0) {
    return conv_forward_gather_scatter_cuda_fallback(in_feat, kernel, neighbor_map,
                                             output_size, conv_mode,
                                             neighbor_offset, transpose);
  } else if (buffer_size * (in_feat.size(1) + kernel.size(-1)) >
                 global_buffer.size(0) &&
             !in_feat.requires_grad()) {
    // std::cout << "fallback: " << buffer_size * (in_feat.size(1) +
    // out_feat.size(1)) << " " << global_buffer.size(0) << std::endl;
    //  global buffer not large enough, fall back
    return conv_forward_gather_scatter_cuda_fallback(in_feat, kernel, neighbor_map,
                                             output_size, conv_mode,
                                             neighbor_offset, transpose);
  } else {
    // std::cout << "not fallback: " << buffer_size * (in_feat.size(1) +
    // out_feat.size(1)) << " " << global_buffer.size(0) << std::endl;
    //  global buffer large enough, do all gather / all scatter
    return conv_forward_gather_scatter_cuda_latest(
        in_feat, kernel, neighbor_map, neighbor_offset, input_mask, output_mask,
        output_size, epsilon, mm_thresh, conv_mode, transpose, global_buffer);
  }
}

void group_strategy_generation(
    // inputs
    int kernel_volume, float epsilon, int mm_thresh, int conv_mode,
    at::Tensor neighbor_offset, bool precompute_mid,
    // outputs
    std::vector<std::vector<int>> &groups, std::vector<int> &mm_ops,
    std::vector<int> &group_sizes, at::Tensor cum_buffer_sizes,
    int &buffer_size) {
  buffer_size = 0;
  bool new_group = true;
  int group_min_size, group_max_size;
  group_min_size = group_max_size = *neighbor_offset.data_ptr<int>();
  std::vector<int> kernel_order;
  std::vector<int> split_points;
  if (conv_mode == 2) {
    for (int i = 0; i < kernel_volume / 2; i++) {
      kernel_order.push_back(i);
      kernel_order.push_back(kernel_volume - 1 - i);
    }
    if (!precompute_mid && kernel_volume % 2 != 0)
      kernel_order.push_back(kernel_volume / 2);
  } else {
    assert(epsilon == 0);
    assert(mm_thresh == 0.0);
    if (precompute_mid && kernel_volume % 2 != 0) {
      for (int i = 0; i < kernel_volume / 2; i++) {
        kernel_order.push_back(i);
      }
      for (int i = kernel_volume / 2 + 1; i < kernel_volume; i++) {
        kernel_order.push_back(i);
      }
    } else {
      for (int i = 0; i < kernel_volume; i++) {
        kernel_order.push_back(i);
      }
    }
  }

  // find split points between groups
  for (int i = 0; i < kernel_order.size(); i++) {
    int cur_size = *(neighbor_offset.data_ptr<int>() + kernel_order[i]);

    if (cur_size < group_min_size) {
      group_min_size = cur_size;
    } else if (cur_size > group_max_size) {
      group_max_size = cur_size;
    }

    if (1.0 - (float)group_min_size / group_max_size > epsilon) {
      // this group ends
      split_points.push_back(i);
      group_min_size = group_max_size =
          *(neighbor_offset.data_ptr<int>() + kernel_order[i]);
    }
  }
  if (split_points.size() == 0 ||
      split_points[split_points.size() - 1] != kernel_order.size())
    split_points.push_back((int)kernel_order.size());

  // determine each group
  for (int i = 0; i < split_points.size(); i++) {
    int cur_split_point = split_points[i];
    int prev_split_point = i > 0 ? split_points[i - 1] : 0;
    std::vector<int> cur_group;
    group_max_size = -1;
    for (int j = prev_split_point; j < cur_split_point; j++) {
      cur_group.push_back(kernel_order[j]);
      group_max_size = std::max(
          group_max_size, *(neighbor_offset.data_ptr<int>() + kernel_order[j]));
    }
    groups.push_back(cur_group);
    group_sizes.push_back(group_max_size);
    if (group_max_size < mm_thresh) {
      // bmm
      mm_ops.push_back(1);
    } else {
      // separate
      mm_ops.push_back(0);
    }
  }

  // determine the cumulated buffer size for each offset
  int cur_cum_size = 0;
  for (int i = 0; i < groups.size(); i++) {
    for (int j = 0; j < groups[i].size(); j++) {
      *(cum_buffer_sizes.data_ptr<int>() + groups[i][j]) = cur_cum_size;
      if (mm_ops[i] == 0) {
        cur_cum_size += *(neighbor_offset.data_ptr<int>() + groups[i][j]);
        buffer_size += *(neighbor_offset.data_ptr<int>() + groups[i][j]);
      } else {
        cur_cum_size += group_sizes[i];
        buffer_size += group_sizes[i];
      }
    }
  }
}

at::Tensor conv_forward_gather_scatter_cuda_latest(
    at::Tensor in_feat, at::Tensor _kernel, at::Tensor neighbor_map,
    at::Tensor neighbor_offset, at::Tensor input_mask, at::Tensor output_mask,
    const int output_size, const float epsilon, const int mm_thresh,
    const int conv_mode, const bool transpose, at::Tensor global_buffer) {
  if (in_feat.size(1) != _kernel.size(1)) {
    throw std::invalid_argument("Input feature size and kernel size mismatch");
  }

  at::Tensor neighbor_offset_gpu = neighbor_offset.to(in_feat.device());
  at::Tensor neighbor_offset_cum_gpu =
      torch::cumsum(neighbor_offset_gpu, 0).to(at::ScalarType::Int);
  at::Tensor neighbor_offset_cum =
      neighbor_offset_cum_gpu.to(neighbor_offset.device());
  at::Tensor cum_buffer_sizes = torch::zeros_like(neighbor_offset);

  auto options =
      torch::TensorOptions().dtype(in_feat.dtype()).device(in_feat.device());
  bool is_half = in_feat.scalar_type() == at::ScalarType::Half;
  at::Tensor out_feat = torch::zeros({output_size, _kernel.size(-1)}, options);

  // pad num channels to an even number
  at::Tensor kernel = _kernel.clone();

  int n_in_channels_original = in_feat.size(1);
  int n_out_channels_original = out_feat.size(1);

  if (is_half) {
    if (in_feat.size(1) % 8 != 0) {
      in_feat = torch::cat(
          {in_feat,
           torch::zeros({in_feat.size(0), 8 - (in_feat.size(1) % 8)}, options)},
          -1);
      kernel = torch::cat(
          {kernel, torch::zeros({kernel.size(0), 8 - (kernel.size(1) % 8),
                                 kernel.size(2)},
                                options)},
          1);
    }
    if (out_feat.size(1) % 8 != 0) {
      out_feat = torch::cat(
          {out_feat,
           torch::zeros({out_feat.size(0), 8 - (out_feat.size(1) % 8)},
                        options)},
          -1);
      kernel = torch::cat({kernel, torch::zeros({kernel.size(0), kernel.size(1),
                                                 8 - (kernel.size(2) % 8)},
                                                options)},
                          -1);
    }
  } else {
    if (in_feat.size(1) % 4 != 0) {
      in_feat = torch::cat(
          {in_feat,
           torch::zeros({in_feat.size(0), 4 - (in_feat.size(1) % 4)}, options)},
          -1);
      kernel = torch::cat(
          {kernel, torch::zeros({kernel.size(0), 4 - (kernel.size(1) % 4),
                                 kernel.size(2)},
                                options)},
          1);
    }
    if (out_feat.size(1) % 4 != 0) {
      out_feat = torch::cat(
          {out_feat,
           torch::zeros({out_feat.size(0), 4 - (out_feat.size(1) % 4)},
                        options)},
          -1);
      kernel = torch::cat({kernel, torch::zeros({kernel.size(0), kernel.size(1),
                                                 4 - (kernel.size(2) % 4)},
                                                options)},
                          -1);
    }
  }

  int n_in_feats = in_feat.size(0);
  int n_in_channels = in_feat.size(1);
  int n_out_feats = out_feat.size(0);
  int n_out_channels = out_feat.size(1);

  int kernel_volume = kernel.size(0);

  // memory optimization
  bool precompute_mid = false;
  // possibly in the last position
  int mid_kernel = conv_mode == 2 ? kernel_volume - 1 : kernel_volume / 2;
  int max_kmap_size = 1;
  // we can precompute features for w[0,0] which avoids gather/scatter
  if (kernel_volume % 2 == 1 && n_in_feats == n_out_feats) {
    precompute_mid = true;
    max_kmap_size =
        *std::max_element(neighbor_offset.data_ptr<int>(),
                          neighbor_offset.data_ptr<int>() + kernel_volume / 2);
    max_kmap_size =
        std::max(max_kmap_size,
                 *std::max_element(
                     neighbor_offset.data_ptr<int>() + kernel_volume / 2 + 1,
                     neighbor_offset.data_ptr<int>() + kernel_volume));
    max_kmap_size = std::max(max_kmap_size, 1);
  } else {
    max_kmap_size =
        *std::max_element(neighbor_offset.data_ptr<int>(),
                          neighbor_offset.data_ptr<int>() + kernel_volume);
  }

  std::vector<std::vector<int>> groups;
  std::vector<int> mm_ops;
  std::vector<int> group_sizes;
  at::Tensor pads, cum_pads;
  int buffer_size;
  // step0.1: get the groups
  group_strategy_generation(kernel_volume, epsilon, mm_thresh, conv_mode,
                            neighbor_offset, precompute_mid, groups, mm_ops,
                            group_sizes, cum_buffer_sizes, buffer_size);
  at::Tensor cum_buffer_sizes_gpu =
      cum_buffer_sizes.to(neighbor_offset_cum_gpu.device());

  // symmetric_mode &= precompute_mid;
  at::Tensor in_buffer, out_buffer;
  if (!in_feat.requires_grad()) {
    if (is_half) {
      in_buffer = torch::from_blob(global_buffer.data_ptr<at::Half>(),
                                   {buffer_size, n_in_channels}, options);
      out_buffer = torch::from_blob(
          global_buffer.data_ptr<at::Half>() + buffer_size * n_in_channels,
          {buffer_size, n_out_channels}, options);
    } else {
      in_buffer = torch::from_blob(global_buffer.data_ptr<float>(),
                                   {buffer_size, n_in_channels}, options);
      out_buffer = torch::from_blob(
          global_buffer.data_ptr<float>() + buffer_size * n_in_channels,
          {buffer_size, n_out_channels}, options);
    }
  } else {
    in_buffer = torch::zeros({buffer_size, n_in_channels}, options);
    out_buffer = torch::zeros({buffer_size, n_out_channels}, options);
  }

  // all gather
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      in_feat.type(), "conv_forward_gather_scatter_cuda", ([&] {
        gather_all_kernel_pad_sep_with_mask<scalar_t>
            <<<ceil((double)(n_in_feats * n_in_channels) /
                    (256 << (sizeof(scalar_t) == 2) + 2)),
               256>>>(n_in_feats, n_in_channels, kernel_volume,
                      in_feat.data_ptr<scalar_t>(),
                      in_buffer.data_ptr<scalar_t>(),
                      neighbor_map.data_ptr<int>(),
                      neighbor_offset_gpu.data_ptr<int>(),
                      neighbor_offset_cum_gpu.data_ptr<int>(),
                      cum_buffer_sizes_gpu.data_ptr<int>(),
                      input_mask.data_ptr<int>(), output_mask.data_ptr<int>(),
                      transpose, precompute_mid);
      }));

  at::Tensor in_buffer_activated, out_buffer_activated, kernel_buffer;
  int buffer_st;
  int cur_buffer_size;
  // for each group
  // mm_ops = 0, sep; mm_ops = 1, BMM
  int kernel_cnt = 0;
  for (int i = 0; i < groups.size(); i++) {
    switch (mm_ops[i]) {
      case 0: {
        for (int j = 0; j < groups[i].size(); j++) {
          int kmap_idx = groups[i][j];
          if (kmap_idx == 0)
            buffer_st = 0;
          else
            buffer_st = *(cum_buffer_sizes.data_ptr<int>() + kmap_idx);
          cur_buffer_size = *(neighbor_offset.data_ptr<int>() + kmap_idx);
          if (is_half) {
            in_buffer_activated = torch::from_blob(
                in_buffer.data_ptr<at::Half>() + buffer_st * n_in_channels,
                {cur_buffer_size, n_in_channels}, options);
            out_buffer_activated = torch::from_blob(
                out_buffer.data_ptr<at::Half>() + buffer_st * n_out_channels,
                {cur_buffer_size, n_out_channels}, options);
          } else {
            in_buffer_activated = torch::from_blob(
                in_buffer.data_ptr<float>() + buffer_st * n_in_channels,
                {cur_buffer_size, n_in_channels}, options);
            out_buffer_activated = torch::from_blob(
                out_buffer.data_ptr<float>() + buffer_st * n_out_channels,
                {cur_buffer_size, n_out_channels}, options);
          }
          if (conv_mode == 2) {
            torch::mm_out(out_buffer_activated, in_buffer_activated,
                          kernel[kernel_cnt]);
            kernel_cnt++;
          } else {
            torch::mm_out(out_buffer_activated, in_buffer_activated,
                          kernel[kmap_idx]);
          }
        }
        break;
      }
      case 1: {
        int kmap_idx = groups[i][0];
        if (kmap_idx == 0)
          buffer_st = 0;
        else
          buffer_st = *(cum_buffer_sizes.data_ptr<int>() + kmap_idx);
        cur_buffer_size = group_sizes[i];
        if (is_half) {
          in_buffer_activated = torch::from_blob(
              in_buffer.data_ptr<at::Half>() + buffer_st * n_in_channels,
              {(int)(groups[i].size()), cur_buffer_size, n_in_channels},
              options);
          out_buffer_activated = torch::from_blob(
              out_buffer.data_ptr<at::Half>() + buffer_st * n_out_channels,
              {(int)(groups[i].size()), cur_buffer_size, n_out_channels},
              options);
          kernel_buffer = torch::from_blob(
              kernel[kernel_cnt].data_ptr<at::Half>(),
              {(int)(groups[i].size()), n_in_channels, n_out_channels},
              options);
        } else {
          in_buffer_activated = torch::from_blob(
              in_buffer.data_ptr<float>() + buffer_st * n_in_channels,
              {(int)(groups[i].size()), cur_buffer_size, n_in_channels},
              options);
          out_buffer_activated = torch::from_blob(
              out_buffer.data_ptr<float>() + buffer_st * n_out_channels,
              {(int)(groups[i].size()), cur_buffer_size, n_out_channels},
              options);
          kernel_buffer = torch::from_blob(
              kernel[kernel_cnt].data_ptr<float>(),
              {(int)(groups[i].size()), n_in_channels, n_out_channels},
              options);
        }
        torch::bmm_out(out_buffer_activated, in_buffer_activated,
                       kernel_buffer);
        kernel_cnt += (int)(groups[i].size());
        break;
      }
    }
  }

  if (is_half) {
    // new version
    scatter_all_kernel_pad_sep_with_mask_half<<<
        ceil((double)(n_out_feats * n_out_channels) / 2048), 256>>>(
        n_out_feats, n_out_channels, kernel_volume,
        reinterpret_cast<half *>(out_buffer.data_ptr<at::Half>()),
        reinterpret_cast<half *>(out_feat.data_ptr<at::Half>()),
        neighbor_map.data_ptr<int>(), neighbor_offset_gpu.data_ptr<int>(),
        neighbor_offset_cum_gpu.data_ptr<int>(),
        cum_buffer_sizes_gpu.data_ptr<int>(), input_mask.data_ptr<int>(),
        output_mask.data_ptr<int>(), transpose, precompute_mid);
  } else {
    // new version
    scatter_all_kernel_pad_sep_with_mask_float<<<
        ceil((double)(n_out_feats * n_out_channels) / 1024), 256>>>(
        n_out_feats, n_out_channels, kernel_volume,
        out_buffer.data_ptr<float>(), out_feat.data_ptr<float>(),
        neighbor_map.data_ptr<int>(), neighbor_offset_gpu.data_ptr<int>(),
        neighbor_offset_cum_gpu.data_ptr<int>(),
        cum_buffer_sizes_gpu.data_ptr<int>(), input_mask.data_ptr<int>(),
        output_mask.data_ptr<int>(), transpose, precompute_mid);
  }

  if (precompute_mid)
    at::addmm_out(out_feat, out_feat, in_feat, kernel[mid_kernel]);

  if (n_out_channels != n_out_channels_original) {
    out_feat = at::slice(out_feat, 1, 0, n_out_channels_original).contiguous();
  }
  return out_feat;
}

at::Tensor conv_forward_gather_scatter_cuda_fallback(
    at::Tensor in_feat, at::Tensor kernel, at::Tensor neighbor_map,
    const int output_size, const int conv_mode, at::Tensor neighbor_offset,
    const bool transpose) {
  if (in_feat.size(1) != kernel.size(1)) {
    throw std::invalid_argument("Input feature size and kernel size mismatch");
  }
  bool is_half = in_feat.scalar_type() == at::ScalarType::Half;
  auto options =
      torch::TensorOptions().dtype(in_feat.dtype()).device(in_feat.device());
  at::Tensor out_feat = torch::zeros({output_size, kernel.size(-1)}, options);

  // need to avoid misaligned memory access
  bool padded = false;
  if (is_half) {
    if (in_feat.size(1) % 2 != 0) {
      in_feat = torch::cat(
          {in_feat, torch::zeros({in_feat.size(0), 1}, options)}, -1);
      kernel = torch::cat(
          {kernel, torch::zeros({kernel.size(0), 1, kernel.size(2)}, options)},
          1);
    }
    if (out_feat.size(1) % 2 != 0) {
      out_feat = torch::cat(
          {out_feat, torch::zeros({out_feat.size(0), 1}, options)}, -1);
      kernel = torch::cat(
          {kernel, torch::zeros({kernel.size(0), kernel.size(1), 1}, options)},
          -1);
      padded = true;
    }
  }

  int n_in_feats = in_feat.size(0);
  int n_in_channels = in_feat.size(1);
  int n_out_feats = out_feat.size(0);
  int n_out_channels = out_feat.size(1);
  int kernel_volume = kernel.size(0);
  // memory optimization
  bool precompute_mid = false;
  int mid_kernel = kernel_volume / 2;
  int in_buffer_size = 1;
  // we can precompute features for w[0,0] which avoids gather/scatter
  if (kernel_volume % 2 == 1 && n_in_feats == n_out_feats) {
    precompute_mid = true;
    in_buffer_size =
        *std::max_element(neighbor_offset.data_ptr<int>(),
                          neighbor_offset.data_ptr<int>() + mid_kernel);
    in_buffer_size = std::max(
        in_buffer_size,
        *std::max_element(neighbor_offset.data_ptr<int>() + mid_kernel + 1,
                          neighbor_offset.data_ptr<int>() + kernel_volume));
    in_buffer_size = std::max(in_buffer_size, 1);
    // (N, c) X (c, o) = (N, o)
    // conv_mode == 2 indicates kernel has been reordered, in which case
    // w[0,0] is placed at the end
    int mid_kmap_idx = conv_mode != 2 ? kernel_volume / 2 : kernel_volume - 1;
    torch::mm_out(out_feat, in_feat, kernel[mid_kmap_idx]);
  } else {
    in_buffer_size =
        *std::max_element(neighbor_offset.data_ptr<int>(),
                          neighbor_offset.data_ptr<int>() + kernel_volume);
  }
  auto in_buffer = torch::zeros({in_buffer_size, n_in_channels}, options);
  auto out_buffer = torch::zeros({in_buffer_size, n_out_channels}, options);
  int cur_offset = 0;
  // gather/gemm/scatter on each weight
  for (int i = 0; i < kernel_volume; i++) {
    int n_active_feats = neighbor_offset.data_ptr<int>()[i];
    // if there's no active features for this weight, skip it
    if (n_active_feats == 0) {
      continue;
    }
    // if w[0,0] was precomputed above, skip it
    if ((i == mid_kernel) && precompute_mid) {
      cur_offset += 2 * n_active_feats;
      continue;
    }
    // in_buffer_activated (i, c) holds the dense input features from gather
    // for i = n_active_feats (# of features in the activated kernel from
    // neighbor_offset) out_buffer_activated (i, o) holds the dense output
    // features to scatter
    at::Tensor out_buffer_activated;
    at::Tensor in_buffer_activated;
    if (is_half) {
      out_buffer_activated =
          torch::from_blob(out_buffer.data_ptr<at::Half>(),
                           {n_active_feats, n_out_channels}, options);
      in_buffer_activated =
          torch::from_blob(in_buffer.data_ptr<at::Half>(),
                           {n_active_feats, n_in_channels}, options);
    } else {
      out_buffer_activated =
          torch::from_blob(out_buffer.data_ptr<float>(),
                           {n_active_feats, n_out_channels}, options);
      in_buffer_activated =
          torch::from_blob(in_buffer.data_ptr<float>(),
                           {n_active_feats, n_in_channels}, options);
    }
    // gather n_active_feats dense features from N sparse input features with c
    // feature dimensions
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        in_feat.type(), "conv_forward_gather_scatter_cuda", ([&] {
          gather_kernel<scalar_t>
              <<<ceil((double)(n_active_feats * n_in_channels) / 256), 256>>>(
                  n_active_feats, n_in_feats, n_in_channels,
                  in_feat.data_ptr<scalar_t>(),
                  in_buffer_activated.data_ptr<scalar_t>(),
                  neighbor_map.data_ptr<int>() + cur_offset, transpose);
        }));
    // gemm: (i, c) X (c, o) = (i, o)
    int kmap_idx = i;
    if (conv_mode == 2) {
      kmap_idx = i < mid_kernel ? i * 2 : (kernel_volume - i) * 2 - 1;
    }
    torch::mm_out(out_buffer_activated, in_buffer_activated, kernel[kmap_idx]);
    // scatter n_active_feats dense features into n_out_feats output features of
    // dimension n_out_channels
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        in_feat.type(), "conv_forward_gather_scatter_cuda", ([&] {
          scatter_kernel<scalar_t>
              <<<ceil((double)(n_active_feats * n_out_channels) / 256), 256>>>(
                  n_active_feats, n_out_feats, n_out_channels,
                  out_buffer_activated.data_ptr<scalar_t>(),
                  out_feat.data_ptr<scalar_t>(),
                  neighbor_map.data_ptr<int>() + cur_offset, transpose);
        }));
    cur_offset += 2 * n_active_feats;
  }

  if (padded) {
    out_feat = at::slice(out_feat, 1, 0, n_out_channels - 1).contiguous();
  }
  return out_feat;
}
void conv_backward_gather_scatter_cuda(at::Tensor in_feat, at::Tensor grad_in_feat,
                               at::Tensor grad_out_feat, at::Tensor kernel,
                               at::Tensor grad_kernel, at::Tensor neighbor_map,
                               at::Tensor neighbor_offset,
                               const bool transpose) {
  grad_in_feat.resize_as_(in_feat);
  grad_in_feat.zero_();
  grad_kernel.resize_as_(kernel);
  grad_kernel.zero_();
  bool is_half = in_feat.scalar_type() == at::ScalarType::Half;
  int n_in_feats = in_feat.size(0);
  int n_in_channels = in_feat.size(1);
  int n_out_feats = grad_out_feat.size(0);
  int n_out_channels = kernel.size(-1);
  int kernel_volume = kernel.size(0);
  bool flag = false;
  int in_buffer_size;
  in_buffer_size =
      *std::max_element(neighbor_offset.data_ptr<int>(),
                        neighbor_offset.data_ptr<int>() + kernel_volume);
  auto options =
      torch::TensorOptions().dtype(in_feat.dtype()).device(in_feat.device());
  auto in_buffer = torch::zeros({in_buffer_size, in_feat.size(1)}, options);
  auto in_grad_buffer =
      torch::zeros({in_buffer_size, in_feat.size(1)}, options);
  auto out_grad_buffer =
      torch::zeros({in_buffer_size, kernel.size(2)}, options);
  int cur_offset = 0;
  for (int i = 0; i < kernel_volume; i++) {
    auto kernel_grad_buffer = grad_kernel[i];
    int n_active_feats = neighbor_offset.data_ptr<int>()[i];
    if (flag && (i == kernel_volume / 2)) {
      cur_offset += 2 * n_active_feats;
      continue;
    }
    if (n_active_feats == 0) {
      continue;
    }
    // Can't figure out a cleaner way to do this
    at::Tensor out_grad_buffer_activated;
    at::Tensor in_grad_buffer_activated;
    at::Tensor in_buffer_activated;
    if (is_half) {
      out_grad_buffer_activated =
          torch::from_blob(out_grad_buffer.data_ptr<at::Half>(),
                           {n_active_feats, kernel.size(2)}, options);
      in_grad_buffer_activated =
          torch::from_blob(in_grad_buffer.data_ptr<at::Half>(),
                           {n_active_feats, in_feat.size(1)}, options);
      in_buffer_activated =
          torch::from_blob(in_buffer.data_ptr<at::Half>(),
                           {n_active_feats, in_feat.size(1)}, options);
    } else {
      out_grad_buffer_activated =
          torch::from_blob(out_grad_buffer.data_ptr<float>(),
                           {n_active_feats, kernel.size(2)}, options);
      in_grad_buffer_activated =
          torch::from_blob(in_grad_buffer.data_ptr<float>(),
                           {n_active_feats, in_feat.size(1)}, options);
      in_buffer_activated =
          torch::from_blob(in_buffer.data_ptr<float>(),
                           {n_active_feats, in_feat.size(1)}, options);
    }
    // gather
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        in_feat.type(), "conv_forward_gather_scatter_cuda", ([&] {
          gather_kernel<scalar_t>
              <<<ceil((double)(n_active_feats * n_out_channels) / 256), 256>>>(
                  n_active_feats, n_out_feats, n_out_channels,
                  grad_out_feat.data_ptr<scalar_t>(),
                  out_grad_buffer_activated.data_ptr<scalar_t>(),
                  neighbor_map.data_ptr<int>() + cur_offset, !transpose);
        }));
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        in_feat.type(), "conv_forward_gather_scatter_cuda", ([&] {
          gather_kernel<scalar_t>
              <<<ceil((double)(n_active_feats * n_in_channels) / 256), 256>>>(
                  n_active_feats, n_in_feats, n_in_channels,
                  in_feat.data_ptr<scalar_t>(),
                  in_buffer_activated.data_ptr<scalar_t>(),
                  neighbor_map.data_ptr<int>() + cur_offset, transpose);
        }));
    // gemm
    torch::mm_out(in_grad_buffer_activated, out_grad_buffer_activated,
                  torch::transpose(kernel[i], 0, 1));
    torch::mm_out(kernel_grad_buffer,
                  torch::transpose(in_buffer_activated, 0, 1),
                  out_grad_buffer_activated);
    // scatter
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        in_feat.type(), "conv_forward_gather_scatter_cuda", ([&] {
          scatter_kernel<scalar_t>
              <<<ceil((double)(n_active_feats * n_in_channels) / 256), 256>>>(
                  n_active_feats, n_in_feats, n_in_channels,
                  in_grad_buffer_activated.data_ptr<scalar_t>(),
                  grad_in_feat.data_ptr<scalar_t>(),
                  neighbor_map.data_ptr<int>() + cur_offset, !transpose);
        }));
    cur_offset += 2 * n_active_feats;
  }
}

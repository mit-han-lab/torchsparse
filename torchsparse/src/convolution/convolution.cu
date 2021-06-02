#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <chrono>
#include <algorithm>
#include "convolution_gpu.h"

template <typename scalar_t>
__global__ void gather_kernel(const int n_k, const int n_in, const int c, 
                               const scalar_t *in_feat, scalar_t *out_feat, const int *kmap,
                               const bool transpose){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int i = index / c;
    int j = index % c;
    if(i >= n_k) return;
    int in_pos = kmap[2 * i + transpose];
    if(in_pos < 0) return;
    out_feat[i * c + j] = in_feat[in_pos * c + j];
}

template <typename scalar_t>
__global__ void scatter_kernel(const int n_in, const int n_out, const int c, 
                               const scalar_t *in_feat, scalar_t *out_feat, const int *kmap,
                               const bool transpose){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int i = index / c;
    int j = index % c;
    if(i >= n_in) return;
    int out_pos = kmap[2 * i + 1 - transpose];
    if(out_pos < 0) return;
    out_feat[out_pos * c + j] += in_feat[i * c + j];
}

// in_feat: (N, c) N=# of input points, c = input channels
// out_feat: (M, o) M=# of output points, o = output channels
//                  for stride=1, M=N. For stride>1, the N input coords
//                  are requantized to M points with grid size (stride * cur_stride)
// kernel: (k^3, c, o) for a 3D convolution of length k
// neighbor_map: (a, 2) the hash table query results from out_coords to in_coords
//                      where neighbor_map[:,0] is the index of the output feature
//                      and neighbor_map[:,1] is the index of the input feature
// neighbor_offset: (k^3) count of active weights based on neighbor_map
//                      with unused weights having 0 and neighbor_offset[k^3/2]
//                      holding w[0,0].
void ConvolutionForwardGPU(at::Tensor in_feat, at::Tensor out_feat,
                           at::Tensor kernel, at::Tensor neighbor_map,
                           at::Tensor neighbor_offset, const bool transpose)
{
    if (in_feat.size(1) != kernel.size(1))
    {
        throw std::invalid_argument("Input feature size and kernel size mismatch");
    }

    bool is_half = in_feat.scalar_type() == at::ScalarType::Half;

    int n_in_feats = in_feat.size(0);
    int n_in_channels = in_feat.size(1);
    int n_out_feats = out_feat.size(0);
    int n_out_channels = out_feat.size(1);;

    int kernel_volume = kernel.size(0);

    // memory optimization
    bool precompute_mid = false;
    int mid_kernel = kernel_volume / 2;
    int in_buffer_size = 1;
    // we can precompute features for w[0,0] which avoids gather/scatter
    if (kernel_volume % 2 == 1 && n_in_feats == n_out_feats)
    {
        precompute_mid = true;
        in_buffer_size = *std::max_element(neighbor_offset.data_ptr<int>(),
                                           neighbor_offset.data_ptr<int>() + mid_kernel);
        in_buffer_size = std::max(in_buffer_size,
                                  *std::max_element(neighbor_offset.data_ptr<int>() + mid_kernel + 1,
                                                    neighbor_offset.data_ptr<int>() + kernel_volume));
        in_buffer_size = std::max(in_buffer_size, 1);

        // (N, c) X (c, o) = (N, o)
        torch::mm_out(out_feat, in_feat, kernel[mid_kernel]);
    }
    else
    {
        in_buffer_size = *std::max_element(neighbor_offset.data_ptr<int>(),
                                           neighbor_offset.data_ptr<int>() + kernel_volume);
    }

    auto options =
        torch::TensorOptions().dtype(in_feat.dtype()).device(in_feat.device());
    auto in_buffer = torch::zeros({in_buffer_size, n_in_channels}, options);
    auto out_buffer = torch::zeros({in_buffer_size, n_out_channels}, options);
    int cur_offset = 0;
    // gather/gemm/scatter on each weight 
    for (int i = 0; i < kernel_volume; i++)
    {        
        int n_active_feats = neighbor_offset.data_ptr<int>()[i];
        // if there's no active features for this weight, skip it
        if (n_active_feats == 0)
        {
            continue;
        }

        // if w[0,0] was precomputed above, skip it
        if ((i == mid_kernel) && precompute_mid)
        {
            cur_offset += 2 * n_active_feats;
            continue;
        }

        // in_buffer_activated (i, c) holds the dense input features from gather
        // for i = n_active_feats (# of features in the activated kernel from neighbor_offset)
        // out_buffer_activated (i, o) holds the dense output features to scatter
        at::Tensor out_buffer_activated;
        at::Tensor in_buffer_activated;
        if (is_half)
        {
            out_buffer_activated =
                torch::from_blob(out_buffer.data_ptr<at::Half>(),
                                 {n_active_feats, n_out_channels}, options);
            in_buffer_activated =
                torch::from_blob(in_buffer.data_ptr<at::Half>(),
                                 {n_active_feats, n_in_channels}, options);
        }
        else
        {
            out_buffer_activated =
                torch::from_blob(out_buffer.data_ptr<float>(),
                                 {n_active_feats, n_out_channels}, options);
            in_buffer_activated =
                torch::from_blob(in_buffer.data_ptr<float>(),
                                 {n_active_feats, n_in_channels}, options);
        }

        // gather n_active_feats dense features from N sparse input features with c feature dimensions
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(in_feat.type(), "ConvolutionForwardGPU", ([&] {
                                        gather_kernel<scalar_t><<<ceil((double)(n_active_feats * n_in_channels) / 256), 256>>>(
                                            n_active_feats,
                                            n_in_feats,
                                            n_in_channels,
                                            in_feat.data_ptr<scalar_t>(),
                                            in_buffer_activated.data_ptr<scalar_t>(),
                                            neighbor_map.data_ptr<int>() + cur_offset,
                                            transpose);
                                    }));

        // gemm: (i, c) X (c, o) = (i, o)
        torch::mm_out(out_buffer_activated, in_buffer_activated, kernel[i]);

        // scatter n_active_feats dense features into n_out_feats output features of dimension n_out_channels
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(in_feat.type(), "ConvolutionForwardGPU", ([&] {
                                        scatter_kernel<scalar_t><<<ceil((double)(n_active_feats * n_out_channels) / 256), 256>>>(
                                            n_active_feats,
                                            n_out_feats,
                                            n_out_channels,
                                            out_buffer_activated.data_ptr<scalar_t>(),
                                            out_feat.data_ptr<scalar_t>(),
                                            neighbor_map.data_ptr<int>() + cur_offset,
                                            transpose);
                                    }));

        cur_offset += 2 * n_active_feats;
    }
}

void ConvolutionBackwardGPU(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor kernel, at::Tensor grad_kernel, at::Tensor neighbor_map,
    at::Tensor neighbor_offset, const bool transpose)
{
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
    in_buffer_size = *std::max_element(neighbor_offset.data_ptr<int>(),
                                       neighbor_offset.data_ptr<int>() + kernel_volume);

    auto options =
        torch::TensorOptions().dtype(in_feat.dtype()).device(in_feat.device());
    auto in_buffer = torch::zeros({in_buffer_size, in_feat.size(1)}, options);
    auto in_grad_buffer = torch::zeros({in_buffer_size, in_feat.size(1)}, options);
    auto out_grad_buffer = torch::zeros({in_buffer_size, kernel.size(2)}, options);

    int cur_offset = 0;
    for (int i = 0; i < kernel_volume; i++)
    {
        auto kernel_grad_buffer = grad_kernel[i];
        int n_active_feats = neighbor_offset.data_ptr<int>()[i];
        if (flag && (i == kernel_volume / 2))
        {
            cur_offset += 2 * n_active_feats;
            continue;
        }

        if (n_active_feats == 0)
        {
            continue;
        }

        // Can't figure out a cleaner way to do this
        at::Tensor out_grad_buffer_activated;
        at::Tensor in_grad_buffer_activated;
        at::Tensor in_buffer_activated;
        if (is_half)
        {
            out_grad_buffer_activated =
                torch::from_blob(out_grad_buffer.data_ptr<at::Half>(),
                                 {n_active_feats, kernel.size(2)}, options);
            in_grad_buffer_activated =
                torch::from_blob(in_grad_buffer.data_ptr<at::Half>(),
                                 {n_active_feats, in_feat.size(1)}, options);
            in_buffer_activated =
                torch::from_blob(in_buffer.data_ptr<at::Half>(),
                                 {n_active_feats, in_feat.size(1)}, options);
        }
        else
        {
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
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(in_feat.type(), "ConvolutionForwardGPU", ([&] {
                                        gather_kernel<scalar_t><<<ceil((double)(n_active_feats * n_out_channels) / 256), 256>>>(
                                            n_active_feats,
                                            n_out_feats,
                                            n_out_channels,
                                            grad_out_feat.data_ptr<scalar_t>(),
                                            out_grad_buffer_activated.data_ptr<scalar_t>(),
                                            neighbor_map.data_ptr<int>() + cur_offset,
                                            !transpose);
                                    }));

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(in_feat.type(), "ConvolutionForwardGPU", ([&] {
                                        gather_kernel<scalar_t><<<ceil((double)(n_active_feats * n_in_channels) / 256), 256>>>(
                                            n_active_feats,
                                            n_in_feats,
                                            n_in_channels,
                                            in_feat.data_ptr<scalar_t>(),
                                            in_buffer_activated.data_ptr<scalar_t>(),
                                            neighbor_map.data_ptr<int>() + cur_offset,
                                            transpose);
                                    }));

        // gemm
        torch::mm_out(in_grad_buffer_activated, out_grad_buffer_activated, torch::transpose(kernel[i], 0, 1));
        torch::mm_out(kernel_grad_buffer, torch::transpose(in_buffer_activated, 0, 1), out_grad_buffer_activated);

        // scatter
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(in_feat.type(), "ConvolutionForwardGPU", ([&] {
                                        scatter_kernel<scalar_t><<<ceil((double)(n_active_feats * n_in_channels) / 256), 256>>>(
                                            n_active_feats,
                                            n_in_feats,
                                            n_in_channels,
                                            in_grad_buffer_activated.data_ptr<scalar_t>(),
                                            grad_in_feat.data_ptr<scalar_t>(),
                                            neighbor_map.data_ptr<int>() + cur_offset,
                                            !transpose);
                                    }));

        cur_offset += 2 * n_active_feats;
    }
}

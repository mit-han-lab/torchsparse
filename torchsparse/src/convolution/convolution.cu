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

void ConvolutionForwardGPU(at::Tensor in_feat, at::Tensor out_feat,
                           at::Tensor kernel, at::Tensor neighbor_map,
                           at::Tensor neighbor_offset, const bool transpose)
{
    if (in_feat.size(1) != kernel.size(1))
    {
        throw std::invalid_argument("Input feature size and kernel size mismatch");
    }

    bool is_half = in_feat.scalar_type() == at::ScalarType::Half;

    int out_nrows = out_feat.size(0);
    out_feat.resize_({out_nrows, kernel.size(2)});
    out_feat.zero_();

    int kernel_volume = kernel.size(0);

    // memory optimization
    bool flag = false;
    int in_buffer_size = 1;
    if (kernel_volume % 2 && out_nrows == in_feat.size(0))
    {
        flag = true;
        in_buffer_size = *std::max_element(neighbor_offset.data_ptr<int>(),
                                           neighbor_offset.data_ptr<int>() + kernel_volume / 2);
        in_buffer_size = std::max(in_buffer_size,
                                  *std::max_element(neighbor_offset.data_ptr<int>() + kernel_volume / 2 + 1,
                                                    neighbor_offset.data_ptr<int>() + kernel_volume));
        in_buffer_size = std::max(in_buffer_size, 1);

        torch::mm_out(out_feat, in_feat, kernel[kernel_volume / 2]);
    }
    else
    {
        in_buffer_size = *std::max_element(neighbor_offset.data_ptr<int>(),
                                           neighbor_offset.data_ptr<int>() + kernel_volume);
    }

    auto options =
        torch::TensorOptions().dtype(in_feat.dtype()).device(in_feat.device());
    auto in_buffer = torch::zeros({in_buffer_size, in_feat.size(1)}, options);
    auto out_buffer = torch::zeros({in_buffer_size, kernel.size(2)}, options);
    int cur_offset = 0;
    for (int i = 0; i < kernel_volume; i++)
    {
        if (flag && (i == kernel_volume / 2))
        {
            cur_offset += 2 * neighbor_offset.data_ptr<int>()[i];
            continue;
        }

        if (neighbor_offset.data_ptr<int>()[i] == 0)
        {
            continue;
        }

        at::Tensor out_buffer_activated;
        at::Tensor in_buffer_activated;
        if (is_half)
        {
            out_buffer_activated =
                torch::from_blob(out_buffer.data_ptr<at::Half>(),
                                 {neighbor_offset.data_ptr<int>()[i], kernel.size(2)}, options);
            in_buffer_activated =
                torch::from_blob(in_buffer.data_ptr<at::Half>(),
                                 {neighbor_offset.data_ptr<int>()[i], in_feat.size(1)}, options);
        }
        else
        {
            out_buffer_activated =
                torch::from_blob(out_buffer.data_ptr<float>(),
                                 {neighbor_offset.data_ptr<int>()[i], kernel.size(2)}, options);
            in_buffer_activated =
                torch::from_blob(in_buffer.data_ptr<float>(),
                                 {neighbor_offset.data_ptr<int>()[i], in_feat.size(1)}, options);
        }

        int n_in = in_buffer_activated.size(0);
        int n_out = in_feat.size(0);
        int c = kernel.size(1);
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(in_feat.type(), "ConvolutionForwardGPU", ([&] {
                                        gather_kernel<scalar_t><<<ceil((double)(n_in * c) / 256), 256>>>(
                                            n_in,
                                            n_out,
                                            c,
                                            in_feat.data_ptr<scalar_t>(),
                                            in_buffer_activated.data_ptr<scalar_t>(),
                                            neighbor_map.data_ptr<int>() + cur_offset,
                                            transpose);
                                    }));

        // gemm
        torch::mm_out(out_buffer_activated, in_buffer_activated, kernel[i]);

        n_in = neighbor_offset.data_ptr<int>()[i];
        n_out = out_nrows;
        c = kernel.size(2);
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(in_feat.type(), "ConvolutionForwardGPU", ([&] {
                                        scatter_kernel<scalar_t><<<ceil((double)(n_in * c) / 256), 256>>>(
                                            neighbor_offset.data_ptr<int>()[i],
                                            out_nrows,
                                            kernel.size(2),
                                            out_buffer_activated.data_ptr<scalar_t>(),
                                            out_feat.data_ptr<scalar_t>(),
                                            neighbor_map.data_ptr<int>() + cur_offset,
                                            transpose);
                                    }));

        cur_offset += 2 * neighbor_offset.data_ptr<int>()[i];
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
        if (flag && (i == kernel_volume / 2))
        {
            cur_offset += 2 * neighbor_offset.data_ptr<int>()[i];
            continue;
        }

        if (neighbor_offset.data_ptr<int>()[i] == 0)
        {
            continue;
        }

        auto out_grad_buffer_activated =
            torch::from_blob(out_grad_buffer.data_ptr<float>(),
                             {neighbor_offset.data_ptr<int>()[i], kernel.size(2)}, options);
        auto in_grad_buffer_activated =
            torch::from_blob(in_grad_buffer.data_ptr<float>(),
                             {neighbor_offset.data_ptr<int>()[i], in_feat.size(1)}, options);
        auto in_buffer_activated =
            torch::from_blob(in_buffer.data_ptr<float>(),
                             {neighbor_offset.data_ptr<int>()[i], in_feat.size(1)}, options);

        // // gather
        // gather_launch(out_grad_buffer_activated.size(0), grad_out_feat.size(0), kernel.size(2),
        //               grad_out_feat.data_ptr<float>(), out_grad_buffer_activated.data_ptr<float>(),
        //               neighbor_map.data_ptr<int>() + cur_offset, !transpose);

        // gather_launch(in_buffer_activated.size(0), in_feat.size(0), kernel.size(1),
        //               in_feat.data_ptr<float>(), in_buffer_activated.data_ptr<float>(),
        //               neighbor_map.data_ptr<int>() + cur_offset, transpose);

        // gemm
        torch::mm_out(in_grad_buffer_activated, out_grad_buffer_activated, torch::transpose(kernel[i], 0, 1));
        torch::mm_out(kernel_grad_buffer, torch::transpose(in_buffer_activated, 0, 1), out_grad_buffer_activated);

        // // scatter
        // scatter_launch(neighbor_offset.data_ptr<int>()[i], in_feat.size(0), kernel.size(1), in_grad_buffer_activated.data_ptr<float>(),
        //                grad_in_feat.data_ptr<float>(), neighbor_map.data_ptr<int>() + cur_offset, !transpose);

        cur_offset += 2 * neighbor_offset.data_ptr<int>()[i];
    }
}

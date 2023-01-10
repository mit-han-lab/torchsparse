# TORCH_VERSION==1.10.0
# CUDA_VERSION==11.3.1
# CUDNN_VERSION==8
# UBUNTU_VERSION==20.04
# PYTHON_VERSION==3.8

FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
##############################################
# You should modify this to match your GPU compute capability
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
##############################################

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV FORCE_CUDA="1"

RUN echo "deb [by-hash=no] http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list \
  && apt-get update \
  && apt-get install \
    --yes \
    --no-install-recommends \
      git \
      ninja-build \
      cmake \
      build-essential \
      libopenblas-dev \
      libsparsehash-dev \
      python3.8-dev \
      python3-pip \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN pip install \
    --find-links https://download.pytorch.org/whl/torch_stable.html \
    --no-cache-dir \
      torch==1.10.0+cu113
      
COPY . .
ENV MAX_JOBS=4
RUN pip install \
    --no-cache-dir \
    --verbose .
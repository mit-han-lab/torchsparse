import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (CUDA_HOME, BuildExtension, CppExtension,
                                       CUDAExtension)

has_cuda = (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv(
    'FORCE_CUDA', '0') == '1'

from torchsparse import __version__

# Notice that CUDA files, header files should not share names with CPP files.
# Otherwise, there will be "ninja: warning: multiple rules generate xxx.o", which leads to
# multiple definitions error!

file_lis = [
    'torchsparse/src/torchsparse_bindings_gpu.cpp',
    'torchsparse/src/convolution/convolution_cpu.cpp',
    'torchsparse/src/convolution/convolution.cu',
    'torchsparse/src/convolution/convolution_gpu.cu',
    'torchsparse/src/hash/hash_cpu.cpp',
    'torchsparse/src/hash/hash.cpp',
    'torchsparse/src/hash/hash_gpu.cu',
    'torchsparse/src/hashmap/hashmap.cu',
    'torchsparse/src/hashmap/hashmap_cpu.cpp',
    'torchsparse/src/interpolation/devox_gpu.cu',
    'torchsparse/src/interpolation/devox_deterministic.cpp',
    'torchsparse/src/interpolation/devox_deterministic_gpu.cu',
    'torchsparse/src/interpolation/devox_cpu.cpp',
    'torchsparse/src/others/convert_neighbor_map.cpp',
    'torchsparse/src/others/convert_neighbor_map_gpu.cu',
    'torchsparse/src/others/convert_neighbor_map_cpu.cpp',
    'torchsparse/src/others/count.cpp',
    'torchsparse/src/others/count_gpu.cu',
    'torchsparse/src/others/count_cpu.cpp',
    'torchsparse/src/others/insertion_gpu.cu',
    'torchsparse/src/others/insertion_cpu.cpp',
    'torchsparse/src/others/query.cpp',
    'torchsparse/src/others/query_cpu.cpp',
] if has_cuda else [
    'torchsparse/src/torchsparse_bindings.cpp',
    'torchsparse/src/convolution/convolution_cpu.cpp',
    'torchsparse/src/hash/hash_cpu.cpp',
    'torchsparse/src/hashmap/hashmap_cpu.cpp',
    'torchsparse/src/interpolation/devox_cpu.cpp',
    'torchsparse/src/others/convert_neighbor_map_cpu.cpp',
    'torchsparse/src/others/insertion_cpu.cpp',
    'torchsparse/src/others/query_cpu.cpp',
    'torchsparse/src/others/count_cpu.cpp'
]

extra_compile_args = {
    'cxx': ['-g', '-O3', '-fopenmp', '-lgomp'],
    'nvcc': ['-O3']
} if has_cuda else {
    'cxx': ['-g', '-O3', '-fopenmp', '-lgomp']
}

extension_type = CUDAExtension if has_cuda else CppExtension
setup(
    name='torchsparse',
    version=__version__,
    packages=find_packages(),
    ext_modules=[
        extension_type('torchsparse_backend',
                       file_lis,
                       extra_compile_args=extra_compile_args)
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)

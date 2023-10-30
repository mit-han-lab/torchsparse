# cython: language_level=3
import glob
import os
import sys

import torch
import torch.cuda
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

# from torchsparse import __version__

from Cython.Build import cythonize

cython_clean_flag = False

version_file = open("./torchsparse/version.py")
version = version_file.read().split("'")[1]
print("torchsparse version:", version)

if (torch.cuda.is_available() and CUDA_HOME is not None) or (
    os.getenv("FORCE_CUDA", "0") == "1"
):
    device = "cuda"
    pybind_fn = f"pybind_{device}.cu"
else:
    device = "cpu"
    pybind_fn = f"pybind_{device}.cpp"

sources = [os.path.join("torchsparse", "backend", pybind_fn)]
for fpath in glob.glob(os.path.join("torchsparse", "backend", "**", "*")):
    if (fpath.endswith("_cpu.cpp") and device in ["cpu", "cuda"]) or (
        fpath.endswith("_cuda.cu") and device == "cuda"
    ):
        sources.append(fpath)

pyx_files = []
for root, dirnames, filenames in os.walk("torchsparse"):
    for filename in filenames:
        file_path = os.path.join(root, filename)
        if file_path.endswith(".py"):
            file_path2 = file_path + "x"
            os.system("mv " + file_path + " " + file_path2)
            os.system("sed -i '1s/^/# cython: language_level=3\\n/' " + file_path2)
            pyx_files.append(file_path2)

if pyx_files == []:
    for root, dirnames, filenames in os.walk("torchsparse"):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            if file_path.endswith(".pyx"):
                pyx_files.append(file_path)

extension_type = CUDAExtension if device == "cuda" else CppExtension
extra_compile_args = {
    "cxx": ["-g", "-O3", "-fopenmp", "-lgomp"],
    "nvcc": ["-O3", "-std=c++17"],
}

setup(
    name="torchsparse",
    version=version,
    packages=find_packages(),
    ext_modules=cythonize(
        [
            extension_type(
                "torchsparse.backend", sources, extra_compile_args=extra_compile_args
            ),
        ]
        + pyx_files
    ),
    install_requires=[
        "numpy",
        "backports.cached_property",
        "tqdm",
        "typing-extensions",
        "wheel",
        "rootpath",
        "attributedict",
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)

# Clean up
if cython_clean_flag:
    for root, dirnames, filenames in os.walk("torchsparse"):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            if file_path.endswith(".c"):
                os.system("rm " + file_path)
            if file_path.endswith(".pyx"):
                os.system("rm " + file_path)

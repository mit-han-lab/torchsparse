import glob
import os
from pathlib import Path
from subprocess import run

import torch
import torch.cuda
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)
from torchsparse.version import __version__

print("torchsparse version:", __version__)

build_ext = BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=True)

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

extension_type = CUDAExtension if device == "cuda" else CppExtension
current_dir = Path(__file__).parent.resolve()
sparsehash_dir = current_dir / "torchsparse" / "backend" / "third_party" / "sparsehash"
sparsehash_dir_inc = sparsehash_dir / "src"
sparseconfig_path = sparsehash_dir_inc / "sparsehash" / "internal" / "sparseconfig.h"

if not sparseconfig_path.exists():
    print("Generating sparseconfig.h ...")
    run(["./configure"], cwd=sparsehash_dir, check=True)
    run(["make", "src/sparsehash/internal/sparseconfig.h"], cwd=sparsehash_dir, check=True)

extra_compile_args = {
    "cxx": ["-O3", "-fopenmp", "-lgomp", f"-I{sparsehash_dir_inc}"],
    "nvcc": ["-O3"],
}

setup(
    name="torchsparse",
    version=__version__,
    packages=find_packages(),
    ext_modules=[
        extension_type(
            "torchsparse.backend", sources, extra_compile_args=extra_compile_args
        )
    ],
    url="https://github.com/mit-han-lab/torchsparse",
    include_package_data=True,
    install_requires=[
        "ninja",
        "numpy",
        "backports.cached_property",
        "tqdm",
        "typing-extensions",
        "wheel",
        "torch",
        "torchvision"
    ],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)

for f in [
    "Makefile",
    "config.log",
    "config.status",
    "src/config.h",
    "src/sparsehash/internal/sparseconfig.h",
    "src/stamp-h1",
]:
    (sparsehash_dir / f).unlink(missing_ok=True)

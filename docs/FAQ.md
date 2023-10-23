## Frequently Asked Questions

Before posting an issue, please go through the following troubleshooting steps on your own:

- Check whether the issue is TorchSparse specific or environment specific. Try creating an isolated environment via Docker or on another computer and see if the error persists. If using TorchSparse as a dependancy of another project, ensure the downstream project is compatible with the version of TorchSparse that you installed.

- Read the error logs line-by-line: if it's a compilation error, the problem will be shown in the log. Often, compilation issues will come from incorrectly configured environment, such as an improper NVCC or PyTorch installation, rather than incompatibility with this library. Please paste the full log message of `pip install -v git+https://github.com/mit-han-lab/torchsparse.git` when you submit the issue.

- Try [completely uninstalling CUDA](https://askubuntu.com/q/530043) and make sure that there are no additional CUDA installations:

  ```bash
  ls /usr/local/cuda* -d
  ```

- Then, follow **all** of the steps for toolkit installation in the [CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html), especially the [post installation actions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions) to set your `LD_LIBRARY_PATH` and `PATH`.

- Ensure that PyTorch and NVCC are using the same version of CUDA:

  ```bash
  nvcc --version
  python -c "import torch; print(torch.version.cuda);"
  ```

- If you're trying to cross-compile the library (i.e. compiling for a different GPU than the one in the system at build time, such as in a docker build), make use of the `TORCH_CUDA_ARCH_LIST` environmental variable. You can use [this chart](http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/) to find your architecture/gencode. For example, if you want to compile for a Turing-architecture GPU, you would do:

  ```bash
  TORCH_CUDA_ARCH_LIST="7.0;7.5" pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git
  ```

- If you see `Killed` in the compilation log, it's likely the compilation failed due to out of memory as a result of parallel compilation. You can limit the number of CPUs the compiler will use by setting the `MAX_JOBS` environmental variable before installation:

  ```bash
  MAX_JOBS=2 pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git
  ```

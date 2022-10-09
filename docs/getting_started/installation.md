# Installation

TorchSparse is available for Python 3.7 to Python 3.10. Before installing torchsparse, make sure that PyTorch has been successfully installed following the [official guide](https://pytorch.org/).

## Installation via Pip Wheels

We provide pre-built torchsparse packages (recommended) with different PyTorch and CUDA versions to simplify the building for the Linux system. 

1. Ensure that at least PyTorch 1.9.0 is installed:

    ```bash
    python -c "import torch; print(torch.__version__)"
    >>> 1.10.0
    ```

2. Find the CUDA version PyTorch was installed with:
    ```bash
    python -c "import torch; print(torch.version.cuda)"
    >>> 11.3
    ```

3. Install TorchSparse:
    ```bash
    pip install torchsparse==2.0.0+torch${TORCH}cu${CUDA} -i https://pypi.hanlab.ai/simple
    ```

    where `${CUDA}` and `${TORCH}` should be replaced by the specific CUDA version and PyTorch version, respectively.
    For example, for PyTorch 1.9.x and CUDA 10.2, type:

    ```bash
    pip install torchsparse==2.0.0+torch19cu102 -i https://pypi.hanlab.ai/simple
    ```

## Installation from Source
You can alternatively choose to install TorchSparse from source:

1. TorchSparse depends on the [Google Sparse Hash](https://github.com/sparsehash/sparsehash) library.

    - On Ubuntu, it can be installed by:

    ```bash
    sudo apt-get install libsparsehash-dev
    ```

    - On Mac OS, it can be installed by:

    ```bash
    brew install google-sparsehash
    ```

    - You can also compile the library locally (if you do not have the sudo permission) and add the library path to the environment variable `CPLUS_INCLUDE_PATH`.

2. Install Python dependencies by:

    ```bash
    pip install -r requirements.txt
    ```

3. Then TorchSparse can be built from source and installed by:

    ```bash
    pip install -e .
    ```
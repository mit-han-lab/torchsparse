# Installation

TorchSparse is available for Python 3.7 to Python 3.10. Before installing torchsparse, make sure that PyTorch has been successfully installed following the [official guide](https://pytorch.org/).

## Installation via Pip Wheels

We provide pre-built torchsparse packages (recommended) with different PyTorch and CUDA versions to simplify the building for the Linux system. 

1. Ensure at least PyTorch 1.9.0 is installed:

    ```bash
    python -c "import torch; print(torch.__version__)"
    >>> 1.10.0
    ```

2. If you want to use TorchSparse with gpus, please ensure PyTorch was installed with CUDA:
    ```bash
    python -c "import torch; print(torch.version.cuda)"
    >>> 11.3
    ```

3. Then the right TorchSparse wheel can be found and installed by running the installation script:

    ```bash
    /bin/bash -c "$(curl -fsSL https://github.com/mit-han-lab/torchsparse/blob/master/install.sh)"
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
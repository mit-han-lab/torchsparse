# TorchSparse

TorchSparse is a high-performance neural network library for point cloud processing.

### [website](http://torchsparse.mit.edu/) | [paper](https://arxiv.org/abs/2204.10319) | [presentation](https://www.youtube.com/watch?v=IIh4EwmcLUs) | [documentation](https://torchsparse.readthedocs.io/en/latest/)

## Installation

The latest released TorchSparse (v2.0.0) can be installed by following the [instructions](https://torchsparse.readthedocs.io/en/latest/getting_started/installation.html) in the documentation.

To install previously released TorchSparse packages (v1.0.0, v1.1.0, v1.2.0, v1.4.0), you can use the command below:

```bash
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@{VERSION}
```

Where `{VERSION}` is the specific TorchSparse version that you want to use (e.g. `v1.4.0`).

If you use TorchSparse in your code, please remember to specify the exact version in your dependencies.

For installation help and troubleshooting, please consult the [Frequently Asked Questions](https://torchsparse.readthedocs.io/en/latest/support/faq.html) before posting an issue.

## Getting Started

### Sparse Tensor

Sparse tensor (`SparseTensor`) is the main data structure for point cloud, which has two data fields:

- Coordinates (`coords`): a 2D integer tensor with a shape of N x 4, where the first three dimensions correspond to quantized x, y, z coordinates, and the last dimension denotes the batch index.
- Features (`feats`): a 2D tensor with a shape of N x C, where C is the number of feature channels.

Most existing datasets provide raw point cloud data with float coordinates. We can use `sparse_quantize` (provided in `torchsparse.utils.quantize`) to voxelize x, y, z coordinates and remove duplicates:

```python
coords -= np.min(coords, axis=0, keepdims=True)
coords, indices = sparse_quantize(coords, voxel_size, return_index=True)
coords = torch.tensor(coords, dtype=torch.int)
feats = torch.tensor(feats[indices], dtype=torch.float)
tensor = SparseTensor(coords=coords, feats=feats)
```

We can then use `sparse_collate_fn` (provided in `torchsparse.utils.collate`) to assemble a batch of `SparseTensor`'s (and add the batch dimension to `coords`). Please refer to [this example](./examples/example.py) for more details.

### Sparse Neural Network

The neural network interface in TorchSparse is very similar to PyTorch:

```python
from torch import nn
from torchsparse import nn as spnn

model = nn.Sequential(
    spnn.Conv3d(in_channels, out_channels, kernel_size),
    spnn.BatchNorm(out_channels),
    spnn.ReLU(True),
)
```

## Citation

If you use TorchSparse in your research, please use the following BibTeX entries:

```bibtex
@inproceedings{tang2022torchsparse,
  title = {{TorchSparse: Efficient Point Cloud Inference Engine}},
  author = {Tang, Haotian and Liu, Zhijian and Li, Xiuyu and Lin, Yujun and Han, Song},
  booktitle = {Conference on Machine Learning and Systems (MLSys)},
  year = {2022}
}
```

```bibtex
@inproceedings{tang2020searching,
  title = {{Searching Efficient 3D Architectures with Sparse Point-Voxel Convolution}},
  author = {Tang, Haotian and Liu, Zhijian and Zhao, Shengyu and Lin, Yujun and Lin, Ji and Wang, Hanrui and Han, Song},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year = {2020}
}
```

## Acknowledgements

TorchSparse is inspired by many existing open-source libraries, including (but not limited to) [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine), [SECOND](https://github.com/traveller59/second.pytorch) and [SparseConvNet](https://github.com/facebookresearch/SparseConvNet).

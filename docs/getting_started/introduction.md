# Introduction

We shortly introduce the fundamental concepts of TorchSparse through self-contained examples.

## Sparse Tensor

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

We can then use `sparse_collate_fn` (provided in `torchsparse.utils.collate`) to assemble a batch of `SparseTensor`'s (and add the batch dimension to `coords`). Please refer to [this example](https://github.com/mit-han-lab/torchsparse/blob/master/examples/example.py) for more details.

## Sparse Neural Network

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

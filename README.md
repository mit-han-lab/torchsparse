# TorchSparse

## News

2020/09/20: We released `torchsparse` v1.1, which is significantly faster than our `torchsparse` v1.0 and is also achieves **1.9x** speedup over [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) v0.5 alpha when running MinkUNet18C!

2020/08/30: We released `torchsparse` v1.0.

## Overview

We release `torchsparse`, a high-performance computing library for efficient 3D sparse convolution. This library aims at accelerating sparse computation in 3D, in particular the Sparse Convolution operation. 

<img src="https://hanlab.mit.edu/projects/spvnas/figures/sparseconv_illustration.gif" width="1080">

The major advantage of this library is that we support all computation on the GPU, especially the kernel map construction (which is done on the CPU in latest [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) V0.4.3).

## Installation

You may run the following command to install torchsparse.

```bash
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git
```

Note that this library depends on Google's [sparse hash map project](https://github.com/sparsehash/sparsehash). In order to install this library, you may run

```bash
sudo apt-get install libsparsehash-dev
```

on Ubuntu servers. If you are not sudo, please clone Google's codebase, compile it and install locally. Finally, add the path to this library to your `CPLUS_INCLUDE_PATH` environmental variable.

For GPU server users, we currently support PyTorch 1.6.0 + CUDA 10.2 + CUDNN 7.6.2. For CPU users, we support PyTorch 1.6.0 (CPU version), MKLDNN backend is optional.

## Usage

Our [SPVNAS](https://github.com/mit-han-lab/e3d) project (ECCV2020) is built with torchsparse. You may navigate to this project and follow the instructions in that codebase to play around.

Here, we also provide a walk-through on some important concepts in torchsparse.

### Sparse Tensor and Point Tensor

In torchsparse, we have two data structures for point cloud storage, namely `torchsparse.SparseTensor` and `torchsparse.PointTensor`. Both structures has two data fields `C` (coordinates) and `F` (features). In `SparseTensor`, we assume that all coordinates are **integer** and **do not duplicate**. However, in `PointTensor`, all coordinates are **floating-point** and can duplicate.

### Sparse Quantize and Sparse Collate

The way to convert a point cloud to `SparseTensor` so that it can be consumed by networks built with Sparse Convolution or Sparse Point-Voxel Convolution is to use the function `torchsparse.utils.sparse_quantize`. An example is given here:

```python
inds, labels, inverse_map = sparse_quantize(pc, feat, labels, return_index=True, return_invs=True)
```

where `pc`, `feat`, `labels` corresponds to point cloud (coordinates, should be integer), feature and ground-truth. The `inds` denotes unique indices in the point cloud coordinates, and `inverse_map` denotes the unique index each point is corresponding to. The `inverse map` is used to restore full point cloud prediction from downsampled prediction.

To combine a list of `SparseTensor`s to a batch, you may want to use the `torchsparse.utils.sparse_collate_fn` function. 

Detailed results are given in [SemanticKITTI dataset preprocessing code](https://github.com/mit-han-lab/e3d/blob/master/spvnas/core/datasets/semantic_kitti.py) in our [SPVNAS](https://github.com/mit-han-lab/e3d) project.

### Computation API

The computation interface in torchsparse is straightforward and very similar to original PyTorch. An example here defines a basic convolution block:

```python
class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, dilation=dilation, stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True)
        )

    def forward(self, x):
        out = self.net(x)
        return out
```

where `spnn`denotes `torchsparse.nn`, and `spnn.Conv3d` means 3D sparse convolution operation, `spnn.BatchNorm` and `spnn.ReLU` denotes 3D sparse tensor batchnorm and activations, respectively. We also support direct convolution kernel call via `torchsparse.nn.functional`, for example:

```python
outputs = torchsparse.nn.functional.conv3d(inputs, kernel, stride=1, dilation=1, transpose=False)
```

where we need to define `inputs`(SparseTensor), `kernel` (of shape k^3 x OC x IC when k > 1,  or OC x IC when k = 1, where k denotes the kernel size and IC, OC means input / output channels). The `outputs` is still a SparseTensor.

Detailed examples are given in [here](https://github.com/mit-han-lab/e3d/blob/master/spvnas/core/modules/dynamic_sparseop.py), where we use the `torchsparse.nn.functional` interfaces to implement weight-shared 3D-NAS modules.

### Sparse Hashmap API

Sparse hash map query is important in 3D sparse computation. It is mainly used to infer a point's memory location (*i.e.* index) given its coordinates. For example, we use this operation in kernel map construction part of 3D sparse convolution, and also sparse voxelization / devoxelization in [Sparse Point-Voxel Convolution](https://arxiv.org/abs/2007.16100). Here, we provide the following example for hash map API:

```python
source_hash = torchsparse.nn.functional.sphash(torch.floor(source_coords).int())
target_hash = torchsparse.nn.functional.sphash(torch.floor(target_coords).int())
idx_query = torchsparse.nn.functional.sphashquery(source_hash, target_hash)
```

In this example, `sphash` is the function converting integer coordinates to hashing. The `sphashquery(source_hash, target_hash)` performs the hash table lookup. Here, the hash map has key `target_hash` and value corresponding to point indices in the target point cloud tensor. For each point in the `source_coords`, we find the point index in `target_coords` which has the same coordinate as it.

### Dummy Training Example

We here provides an entire training example with dummy input [here](examples/example.py). In this example, we cover 

- How we start from point cloud data and convert it to SparseTensor format;
- How we can implement SparseTensor batching;
- How to train a semantic segmentation SparseConvNet.

You are also welcomed to check out our [SPVNAS](https://github.com/mit-han-lab/e3d) project to implement training / inference with real data.

### Mixed Precision (float16) Support

Mixed precision training is supported via `torch.cuda.amp.autocast` and `torch.cuda.amp.GradScaler`. Enabling mixed precision training can speed up training and reduce GPU memory usage. By wrapping your training code in a `torch.cuda.amp.autocast` block, feature tensors will automatically be converted to float16 if possible. See [here](examples/example.py) for a complete example. 

## Speed Comparison Between torchsparse and MinkowskiEngine

We benchmark the performance of our torchsparse and latest [MinkowskiEngine V0.4.3](https://github.com/NVIDIA/MinkowskiEngine) here, latency is measured on NVIDIA GTX 1080Ti GPU:

|         Network          | Latency (ME V0.4.3) | Latency (torchsparse V1.0.0) |
| :----------------------: | :-----------------: | :--------------------------: |
| MinkUNet18C (MACs / 10)  |        224.7        |            124.3             |
|  MinkUNet18C (MACs / 4)  |        244.3        |            160.9             |
| MinkUNet18C (MACs / 2.5) |        269.6        |            214.3             |
|       MinkUNet18C        |        323.5        |            294.0             |

## Citation

If you find this code useful, please consider citing:

```bibtex
@inproceedings{
  tang2020searching,
  title = {Searching Efficient 3D Architectures with Sparse Point-Voxel Convolution},
  author = {Tang, Haotian* and Liu, Zhijian* and Zhao, Shengyu and Lin, Yujun and Lin, Ji and Wang, Hanrui and Han, Song},
  booktitle = {European Conference on Computer Vision},
  year = {2020}
}
```

## Acknowledgements

This library is inspired by [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine), [SECOND](https://github.com/traveller59/second.pytorch) and [SparseConvNet](https://github.com/facebookresearch/SparseConvNet).

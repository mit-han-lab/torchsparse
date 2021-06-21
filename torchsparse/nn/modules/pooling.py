from torch import nn

from torchsparse.sparse_tensor import SparseTensor

from torchsparse.nn import functional as spF

__all__ = ['GlobalAveragePooling', 'GlobalMaxPooling']


class GlobalAveragePooling(nn.Module):
    def forward(self, inputs):
        return spF.global_avg_pool(inputs)


class GlobalMaxPooling(nn.Module):
    def forward(self, inputs):
        return spF.global_max_pool(inputs)

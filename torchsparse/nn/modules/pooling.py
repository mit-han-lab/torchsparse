from torch import nn

from ..functional import global_avg_pool, global_max_pool

__all__ = ['GlobalAvgPool', 'GlobalMaxPool']


class GlobalAvgPool(nn.Module):
    def forward(self, inputs):
        return global_avg_pool(inputs)


class GlobalMaxPool(nn.Module):
    def forward(self, inputs):
        return global_max_pool(inputs)

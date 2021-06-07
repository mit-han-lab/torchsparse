import numpy as np
import torch

__all__ = ['SparseTensor']


class SparseTensor:
    def __init__(self, feats, coords, stride=1):
        self.F = feats
        self.C = coords
        self.s = stride
        self.coord_maps = {}
        self.kernel_maps = {}

    def check(self):
        if self.s not in self.coord_maps:
            if isinstance(self.s, int):
                self.coord_maps[self.s] = self.C
            else:
                assert isinstance(self.s, torch.Tensor)
                if len(self.s.shape) > 1:
                    key = str(self.s[0].cpu().int().numpy())
                else:
                    key = str(self.s.cpu().int().numpy())
                self.coord_maps[key] = self.C

    def cuda(self):
        assert type(self.F) == torch.Tensor
        assert type(self.C) == torch.Tensor
        self.F = self.F.cuda()
        self.C = self.C.cuda()
        return self

    def detach(self):
        assert type(self.F) == torch.Tensor
        assert type(self.C) == torch.Tensor
        self.F = self.F.detach()
        self.C = self.C.detach()
        return self

    def to(self, device, non_blocking=True):
        assert type(self.F) == torch.Tensor
        assert type(self.C) == torch.Tensor
        self.F = self.F.to(device, non_blocking=non_blocking)
        self.C = self.C.to(device, non_blocking=non_blocking)
        return self

    def __add__(self, other):
        tensor = SparseTensor(self.F + other.F, self.C, self.s)
        tensor.coord_maps = self.coord_maps
        tensor.kernel_maps = self.kernel_maps
        return tensor

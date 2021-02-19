import torch

__all__ = ['SparseTensor']


class SparseTensor:
    def __init__(self, feats, coords, stride: int = 1) -> None:
        assert type(feats) == torch.Tensor
        assert type(coords) == torch.Tensor
        self.feats = feats
        self.coords = coords
        self.stride = stride
        self.coord_maps = dict()
        self.kernel_maps = dict()

    @property
    def F(self):
        return self.feats

    @property
    def C(self):
        return self.coords

    @property
    def s(self):
        return self.stride

    def check(self):
        if self.s not in self.coord_maps:
            self.coord_maps[self.s] = self.C

    def cuda(self):
        self.feats = self.feats.cuda()
        self.coords = self.coords.cuda()
        return self

    def detach(self):
        self.feats = self.feats.detach()
        self.coords = self.coords.detach()
        return self

    def to(self, device, non_blocking: bool = True):
        self.feats = self.feats.to(device, non_blocking=non_blocking)
        self.coords = self.coords.to(device, non_blocking=non_blocking)
        return self

    def __add__(self, other):
        tensor = SparseTensor(coords=self.coords,
                              feats=self.feats + other.feats,
                              stride=self.stride)
        tensor.coord_maps = self.coord_maps
        tensor.kernel_maps = self.kernel_maps
        return tensor

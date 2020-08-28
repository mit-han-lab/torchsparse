import torch

__all__ = ['PointTensor']


class PointTensor:
    def __init__(self, feat, coords, idx_query=None, weights=None):
        self.F = feat
        self.C = coords
        self.idx_query = idx_query if idx_query is not None else {}
        self.weights = weights if weights is not None else {}
        self.additional_features = {}
        self.additional_features['idx_query'] = {}
        self.additional_features['counts'] = {}

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
        tensor = PointTensor(self.F + other.F, self.C, self.idx_query,
                             self.weights)
        tensor.additional_features = self.additional_features
        return tensor

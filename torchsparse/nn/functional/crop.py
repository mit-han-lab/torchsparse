from torchsparse.sparse_tensor import SparseTensor

__all__ = ['spcrop']


def spcrop(inputs: SparseTensor, loc_min, loc_max) -> SparseTensor:
    coords, feats, stride = inputs.C, inputs.F, inputs.s
    mask = ((coords[:, :3] >= loc_min) & (coords[:, :3] < loc_max)).all(-1)
    coords, feats = coords[mask], feats[mask]
    return SparseTensor(coords=coords, feats=feats, stride=stride)

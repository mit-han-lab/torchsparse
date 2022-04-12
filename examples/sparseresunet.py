import numpy as np
import torch
from models import SparseResUNet

from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = SparseResUNet(cr=1.0).to(device).eval()
    print(model)

    # generate data
    input_size, voxel_size = 10000, 0.2
    inputs = np.random.uniform(-100, 100, size=(input_size, 4))
    pcs, feats = inputs[:, :3], inputs
    pcs -= np.min(pcs, axis=0, keepdims=True)
    pcs, indices = sparse_quantize(pcs, voxel_size, return_index=True)
    coords = np.zeros((pcs.shape[0], 4))
    coords[:, :3] = pcs[:, :3]
    coords[:, -1] = 0
    coords = torch.as_tensor(coords, dtype=torch.int)
    feats = torch.as_tensor(feats[indices], dtype=torch.float)
    input = SparseTensor(coords=coords, feats=feats).to(device)

    # forward
    feats_dict = model(input)
    print(f"output (up4) feature shape {feats_dict['out'].feats.shape}")
    print(f"downsample stage1 feature shape {feats_dict['stage1'].feats.shape}")
    print(f"upsample up1 feature shape {feats_dict['up1'].feats.shape}")

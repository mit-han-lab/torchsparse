import numpy as np
import torch

from torchsparse import SparseTensor
from torchsparse.backbones import SparseResNet18, SparseResUNet18
from torchsparse.utils.quantize import sparse_quantize


@torch.no_grad()
def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    for backbone in [SparseResNet18, SparseResUNet18]:
        print(f'Running model {backbone.__name__}')
        model = backbone(in_channels=4, width_multiplier=1.0).to(device).eval()

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

        # print feature shapes
        for key in feats_dict:
            print(f'{key} feature shape {feats_dict[key].feats.shape}')


if __name__ == '__main__':
    main()

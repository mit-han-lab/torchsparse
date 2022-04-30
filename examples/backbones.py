import numpy as np
import torch
from torch import nn

from torchsparse import SparseTensor
from torchsparse.backbones import SparseResNet21D, SparseResUNet42
from torchsparse.utils.quantize import sparse_quantize


@torch.no_grad()
def main() -> None:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    for backbone in [SparseResNet21D, SparseResUNet42]:
        print(f'{backbone.__name__}:')
        model: nn.Module = backbone(in_channels=4, width_multiplier=1.0)
        model = model.to(device).eval()

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
        outputs = model(input)

        # print feature shapes
        for k, output in enumerate(outputs):
            print(f'output[{k}].F.shape = {output.feats.shape}')


if __name__ == '__main__':
    main()

import argparse
import random
from typing import Any, Dict

import numpy as np
import torch
import torch.utils.data
from torch import nn
from torch.cuda import amp

from torchsparse import SparseTensor
from torchsparse import nn as spnn
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize


class RandomDataset:

    def __init__(self, input_size: int, voxel_size: float) -> None:
        self.input_size = input_size
        self.voxel_size = voxel_size

    def __getitem__(self, _: int) -> Dict[str, Any]:
        inputs = np.random.uniform(-100, 100, size=(self.input_size, 4))
        labels = np.random.choice(10, size=self.input_size)

        coords, feats = inputs[:, :3], inputs
        coords -= np.min(coords, axis=0, keepdims=True)
        coords, indices = sparse_quantize(coords,
                                          self.voxel_size,
                                          return_index=True)

        coords = torch.tensor(coords, dtype=torch.int)
        feats = torch.tensor(feats[indices], dtype=torch.float)
        labels = torch.tensor(labels[indices], dtype=torch.long)

        input = SparseTensor(coords=coords, feats=feats)
        label = SparseTensor(coords=coords, feats=labels)
        return {'input': input, 'label': label}

    def __len__(self):
        return 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--amp_enabled', action='store_true')
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    dataset = RandomDataset(input_size=10000, voxel_size=0.2)
    dataflow = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        collate_fn=sparse_collate_fn,
    )

    model = nn.Sequential(
        spnn.Conv3d(4, 32, 3),
        spnn.BatchNorm(32),
        spnn.ReLU(True),
        spnn.Conv3d(32, 64, 2, stride=2),
        spnn.BatchNorm(64),
        spnn.ReLU(True),
        spnn.Conv3d(64, 64, 2, stride=2, transposed=True),
        spnn.BatchNorm(64),
        spnn.ReLU(True),
        spnn.Conv3d(64, 32, 3),
        spnn.BatchNorm(32),
        spnn.ReLU(True),
        spnn.Conv3d(32, 10, 1),
    ).to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = amp.GradScaler(enabled=args.amp_enabled)

    for k, feed_dict in enumerate(dataflow):
        inputs = feed_dict['input'].to(device=args.device)
        labels = feed_dict['label'].to(device=args.device)

        with amp.autocast(enabled=args.amp_enabled):
            outputs = model(inputs)
            loss = criterion(outputs.feats, labels.feats)

        print(f'[step {k + 1}] loss = {loss.item()}')

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

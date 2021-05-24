import numpy as np
import torch
import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn
from torchsparse import SparseTensor
from torchsparse.utils import sparse_collate_fn, sparse_quantize
import argparse


def generate_random_point_cloud(size=100000, voxel_size=0.2):
    pc = np.random.randn(size, 4)
    pc[:, :3] = pc[:, :3] * 10
    rounded_pc = np.round(pc[:, :3] / voxel_size).astype(np.int32)
    labels = np.random.choice(10, size)
    inds, _, inverse_map = sparse_quantize(rounded_pc,
                                           pc,
                                           labels,
                                           return_index=True,
                                           return_invs=True)

    voxel_pc = rounded_pc[inds]
    voxel_feat = pc[inds]
    voxel_labels = labels[inds]

    sparse_tensor = SparseTensor(voxel_feat, voxel_pc)
    label_tensor = SparseTensor(voxel_labels, voxel_pc)

    feed_dict = {'lidar': sparse_tensor, 'targets': label_tensor}

    return feed_dict


def generate_batched_random_point_clouds(size=100000,
                                         voxel_size=0.2,
                                         batch_size=2):
    batch = []
    for i in range(batch_size):
        batch.append(generate_random_point_cloud(size, voxel_size))
    return sparse_collate_fn(batch)


def dummy_train(device, mixed=False):
    model = nn.Sequential(
        spnn.Conv3d(4, 32, kernel_size=3, stride=1), spnn.BatchNorm(32),
        spnn.ReLU(True), spnn.Conv3d(32, 64, kernel_size=2, stride=2),
        spnn.BatchNorm(64), spnn.ReLU(True),
        spnn.Conv3d(64, 64, kernel_size=2, stride=2, transpose=True),
        spnn.BatchNorm(64), spnn.ReLU(True),
        spnn.Conv3d(64, 32, kernel_size=3, stride=1), spnn.BatchNorm(32),
        spnn.ReLU(True), spnn.Conv3d(32, 10, kernel_size=1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss().to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=mixed)

    print('Starting dummy training...')
    for i in range(10):
        optimizer.zero_grad()
        feed_dict = generate_batched_random_point_clouds()
        inputs = feed_dict['lidar'].to(device)
        targets = feed_dict['targets'].F.to(device).long()
        with torch.cuda.amp.autocast(enabled=mixed):
            outputs = model(inputs)
            loss = criterion(outputs.F, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        print('[step %d] loss = %f.' % (i, loss.item()))
    print('Finished dummy training!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixed", action="store_true")
    args = parser.parse_args()

    # set seeds for reproducibility
    np.random.seed(2021)
    torch.manual_seed(2021)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dummy_train(device, args.mixed)
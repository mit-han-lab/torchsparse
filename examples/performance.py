from datetime import datetime

import numpy as np
import torch
import torch.autograd.profiler as profiler
import torch.cuda
import torch.nn as nn
import torch.optim

import torchsparse.nn as spnn
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize


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
    for _ in range(batch_size):
        batch.append(generate_random_point_cloud(size, voxel_size))
    return sparse_collate_fn(batch)


def dummy_train_3x3(device):
    model = nn.Sequential(
        spnn.Conv3d(4, 32, kernel_size=3, stride=1),
        spnn.Conv3d(32, 64, kernel_size=3, stride=1),
        spnn.Conv3d(64, 128, kernel_size=3, stride=1),
        spnn.Conv3d(128, 256, kernel_size=3, stride=1),
        spnn.Conv3d(256, 128, kernel_size=3, stride=1, transposed=True),
        spnn.Conv3d(128, 64, kernel_size=3, stride=1, transposed=True),
        spnn.Conv3d(64, 32, kernel_size=3, stride=1, transposed=True),
        spnn.Conv3d(32, 10, kernel_size=3, stride=1, transposed=True),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss().to(device)

    print('Starting dummy_train_3x3...')
    time = datetime.now()
    with profiler.profile(profile_memory=True, use_cuda=True) as prof:
        with profiler.record_function('model_inference'):
            for _ in range(10):
                feed_dict = generate_batched_random_point_clouds()
                inputs = feed_dict['lidar'].to(device)
                targets = feed_dict['targets'].F.to(device).long()
                outputs = model(inputs)
                optimizer.zero_grad()
                loss = criterion(outputs.F, targets)
                loss.backward()
                optimizer.step()
                # print('[step %d] loss = %f.'%(i, loss.item()))
    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
    prof.export_chrome_trace('trace_dummy_3x3.json')

    time = datetime.now() - time
    print('Finished dummy_train_3x3 in ', time)


def dummy_train_3x1(device):
    model = nn.Sequential(
        spnn.Conv3d(4, 32, kernel_size=(3, 1, 3), stride=1),
        spnn.Conv3d(32, 64, kernel_size=(1, 3, 3), stride=1),
        spnn.Conv3d(64, 128, kernel_size=(3, 1, 3), stride=1),
        spnn.Conv3d(128, 256, kernel_size=(1, 3, 3), stride=1),
        spnn.Conv3d(256, 128, kernel_size=(3, 1, 3), stride=1, transposed=True),
        spnn.Conv3d(128, 64, kernel_size=(1, 3, 3), stride=1, transposed=True),
        spnn.Conv3d(64, 32, kernel_size=(3, 1, 3), stride=1, transposed=True),
        spnn.Conv3d(32, 10, kernel_size=(1, 3, 3), stride=1, transposed=True),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss().to(device)

    print('Starting dummy_train_3x1 ...')
    time = datetime.now()
    with profiler.profile(profile_memory=True, use_cuda=True) as prof:
        with profiler.record_function('model_inference'):
            for _ in range(10):
                feed_dict = generate_batched_random_point_clouds()
                inputs = feed_dict['lidar'].to(device)
                targets = feed_dict['targets'].F.to(device).long()
                outputs = model(inputs)
                optimizer.zero_grad()
                loss = criterion(outputs.F, targets)
                loss.backward()
                optimizer.step()
                # print('[step %d] loss = %f.'%(i, loss.item()))
    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
    prof.export_chrome_trace('trace_dummy_3x1.json')

    time = datetime.now() - time
    print('Finished dummy_train_3x1 in ', time)


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    dummy_train_3x1(device)
    dummy_train_3x3(device)

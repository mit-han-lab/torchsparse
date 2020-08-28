import numpy as np

import torch
import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn
from torchsparse import SparseTensor
from torchsparse.utils import sparse_quantize, sparse_collate_fn


def generate_random_point_cloud(size=100000, voxel_size=0.2):
    pc = np.random.randn(size, 4)
    pc[:, :3] = pc[:, :3] * 10
    rounded_pc = np.round(pc[:, :3] / voxel_size).astype(np.int32)
    labels = np.random.choice(10, size)
    inds, _, inverse_map = sparse_quantize(
        rounded_pc,
        pc,
        labels,
        return_index=True,
        return_invs=True
    )
    
    voxel_pc = rounded_pc[inds]
    voxel_feat = pc[inds]
    voxel_labels = labels[inds]
    
    sparse_tensor = SparseTensor(voxel_feat, voxel_pc)
    label_tensor = SparseTensor(voxel_labels, voxel_pc)
    
    feed_dict = {
        'lidar': sparse_tensor,
        'targets': label_tensor
    }
    
    return feed_dict


def generate_batched_random_point_clouds(
    size=100000, 
    voxel_size=0.2,
    batch_size=2
):
    batch = []
    for i in range(batch_size):
        batch.append(generate_random_point_cloud(size, 
                                                 voxel_size))
    return sparse_collate_fn(batch)


def dummy_train(device):
    model = nn.Sequential(
        spnn.Conv3d(4, 32, kernel_size=3, stride=1),
        spnn.BatchNorm(32),
        spnn.ReLU(True),
        
        spnn.Conv3d(32, 64, kernel_size=2, stride=2),
        spnn.BatchNorm(64),
        spnn.ReLU(True),
        
        spnn.Conv3d(64, 64, kernel_size=2, stride=2, transpose=True),
        spnn.BatchNorm(64),
        spnn.ReLU(True),
        
        spnn.Conv3d(64, 32, kernel_size=3, stride=1),
        spnn.BatchNorm(32),
        spnn.ReLU(True),
        
        spnn.Conv3d(32, 10, kernel_size=1)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss().to(device)
    
    print('Starting dummy training...')
    for i in range(10):
        feed_dict = generate_batched_random_point_clouds()
        inputs = feed_dict['lidar'].to(device)
        targets = feed_dict['targets'].F.to(device).long()
        outputs = model(inputs)
        optimizer.zero_grad()
        loss = criterion(outputs.F, targets)
        loss.backward()
        optimizer.step()
        print('[step %d] loss = %f.'%(i, loss.item()))
    print('Finished dummy training!')
    

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dummy_train(device)
    

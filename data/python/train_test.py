import h5py
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.preprocessing import StandardScaler
import copy
import os.path

from torch.utils.data import Dataset, DataLoader
import argparse

import torch
import torch.utils.data
from torch import nn
from torch.cuda import amp

import torchsparse
from torchsparse import SparseTensor
from torchsparse import nn as spnn
from torchsparse.utils.collate import sparse_collate, sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize
from torch.cuda import amp
from typing import Any, Dict

class CustomDataset(Dataset):
    def __init__(self, coords, feats, labels):
        coords = torch.tensor(coords, dtype=torch.int)

        
        
        feats = torch.tensor(feats, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)

        self.coords = coords
        self.feats = feats
        self.labels = labels
    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        return self.coords[idx], self.feats[idx], self.labels[idx]

if __name__ == '__main__':
    #load data
    ISOTOPE = "Mg22"
    coords_train = np.load('../mg22simulated/' + ISOTOPE + "_coords_train.npy")
    coords_val = np.load('../mg22simulated/' + ISOTOPE + "_coords_val.npy")
    coords_test = np.load('../mg22simulated/' + ISOTOPE + "_coords_test.npy")
    feats_train = np.load('../mg22simulated/' + ISOTOPE + "_feats_train.npy")
    feats_val = np.load('../mg22simulated/' + ISOTOPE + "_feats_val.npy")
    feats_test = np.load('../mg22simulated/' + ISOTOPE + "_feats_test.npy")
    labels_train = np.load('../mg22simulated/' + ISOTOPE + "_labels_train.npy")
    labels_val = np.load('../mg22simulated/' + ISOTOPE + "_labels_val.npy")
    labels_test = np.load('../mg22simulated/' + ISOTOPE + "_labels_test.npy")
    
    # GPU Settings
    device = 'cuda'
    amp_enabled = True
    
    model = nn.Sequential(
        spnn.Conv3d(4, 32, 3),
        spnn.BatchNorm(32),
        spnn.ReLU(True),
        spnn.Conv3d(32, 32, 3),
        spnn.BatchNorm(32),
        spnn.ReLU(True),
        spnn.Conv3d(32, 5, 1),
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = amp.GradScaler(enabled=amp_enabled)
    
    custom_dataset = CustomDataset(coords_train, feats_train, labels_train)
    
    batch_size = 4
    
    data_loader = DataLoader(
        custom_dataset,
        batch_size=batch_size
    )
    
    for batch_idx, (batch_coords, batch_feats, batch_labels) in enumerate(data_loader):
    
        inputs_list = []
        labels_list = []
        
        for i in range(len(batch_coords)):
            inputs_sparse = SparseTensor(coords=batch_coords[i], feats=batch_feats[i])
            labels_sparse = SparseTensor(coords=batch_coords[i], feats=batch_labels[i])
            inputs_list.append(inputs_sparse)
            labels_list.append(labels_sparse)
            
        inputs = sparse_collate(inputs_list).to(device=device)
        labels = sparse_collate(labels_list).to(device=device)
    
        print(f"Batch {batch_idx + 1}:")
        
        with amp.autocast(enabled=amp_enabled):
            outputs = model(inputs)
            labelsloss = labels.feats.squeeze(-1)
            loss = criterion(outputs.feats, labelsloss)
        
        print(f'[step {batch_idx + 1}] loss = {loss.item()}')
    
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    

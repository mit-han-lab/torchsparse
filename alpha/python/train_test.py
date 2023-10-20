"""
Script Name: train_test.py

Description:
    THIS SCRIPT WAS PURELY FOR TESTING A TRAINING PIPELINE.
    You may reference this code to adapt to your model, but an existing working version exists.

Notes:
    - The data is loaded from the `mg22simulated` directory. 
    - The data has already been processed with the data_processing.sh shell script.

Author: Ian Heung
Date: 10/19/2023

"""


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.cuda import amp
from torchsparse import SparseTensor, nn as spnn
from torchsparse.utils.collate import sparse_collate

class CustomDataset(Dataset):
    """Custom dataset for handling 3D coordinates, features, and labels."""
    
    def __init__(self, coords, feats, labels):
        """Initialize the dataset with coordinates, features, and labels."""
        self.coords = torch.tensor(coords, dtype=torch.int)
        self.feats = torch.tensor(feats, dtype=torch.float)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.coords)

    def __getitem__(self, idx):
        """Fetch a sample given its index."""
        return self.coords[idx], self.feats[idx], self.labels[idx]

if __name__ == '__main__':
    # Load data
    ISOTOPE = "Mg22"
    path = '../mg22simulated/'
    
    coords_train = np.load(path + ISOTOPE + "_coords_train.npy")
    feats_train = np.load(path + ISOTOPE + "_feats_train.npy")
    labels_train = np.load(path + ISOTOPE + "_labels_train.npy")
    
    # Set GPU settings
    device = 'cuda'
    amp_enabled = True
    
    # Define the model architecture
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
    
    # Create DataLoader
    dataset = CustomDataset(coords_train, feats_train, labels_train)
    batch_size = 4
    data_loader = DataLoader(dataset, batch_size=batch_size)
    
    # Training loop
    for batch_idx, (batch_coords, batch_feats, batch_labels) in enumerate(data_loader):
        inputs_list, labels_list = [], []
        
        # Convert to SparseTensors
        for i in range(len(batch_coords)):
            inputs_sparse = SparseTensor(coords=batch_coords[i], feats=batch_feats[i])
            labels_sparse = SparseTensor(coords=batch_coords[i], feats=batch_labels[i])
            inputs_list.append(inputs_sparse)
            labels_list.append(labels_sparse)
            
        inputs = sparse_collate(inputs_list).to(device=device)
        labels = sparse_collate(labels_list).to(device=device)
    
        with amp.autocast(enabled=amp_enabled):
            outputs = model(inputs)
            labelsloss = labels.feats.squeeze(-1)
            loss = criterion(outputs.feats, labelsloss)
        
        print(f'[step {batch_idx + 1}] loss = {loss.item()}')
    
        # Update model weights
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
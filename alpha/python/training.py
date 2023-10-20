import h5py
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.preprocessing import StandardScaler
import copy

import os
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
from sklearn.metrics import confusion_matrix
import datetime
import sys

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
    if len(sys.argv) > 1:
        datetime_str = sys.argv[1]
        print(f"Received datetime: {datetime_str}")
    else:
        print("No datetime argument provided!")
    
    ISOTOPE = "Mg22"
    coords_train = np.load('../mg22simulated/' + ISOTOPE + "_coords_train.npy")
    coords_val = np.load('../mg22simulated/' + ISOTOPE + "_coords_val.npy")
    feats_train = np.load('../mg22simulated/' + ISOTOPE + "_feats_train.npy")
    feats_val = np.load('../mg22simulated/' + ISOTOPE + "_feats_val.npy")
    labels_train = np.load('../mg22simulated/' + ISOTOPE + "_labels_train.npy")
    labels_val = np.load('../mg22simulated/' + ISOTOPE + "_labels_val.npy")
    
    
    coords_train = coords_train[0:100]
    feats_train = feats_train[0:100]
    labels_train = labels_train[0:100]
    
    coords_val = coords_val[0:50]
    feats_val = feats_val[0:50]
    labels_val = labels_val[0:50]
    


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
    
    lr = 1e-3
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = amp.GradScaler(enabled=amp_enabled)


    num_epochs = 100
    batch_size = 12
    
    train_set = CustomDataset(coords_train, feats_train, labels_train)
    train_loader = DataLoader(train_set, batch_size=batch_size)
    
    val_set = CustomDataset(coords_val, feats_val, labels_val)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    
    train_steps = len(train_loader)
    val_steps = len(val_loader)
    
    training_losses = []
    validation_losses = []
    
    for epoch in range(num_epochs):
    
        model.train()
        running_loss = 0.0
        
        for batch_idx, (batch_coords, batch_feats, batch_labels) in enumerate(train_loader):
            
            tr_inputs_list = []
            tr_labels_list = []
        
            for i in range(len(batch_coords)):
                inputs_sparse = SparseTensor(coords=batch_coords[i], feats=batch_feats[i])
                labels_sparse = SparseTensor(coords=batch_coords[i], feats=batch_labels[i])
                tr_inputs_list.append(inputs_sparse)
                tr_labels_list.append(labels_sparse)
            
            tr_inputs = sparse_collate(tr_inputs_list).to(device=device)
            tr_labels = sparse_collate(tr_labels_list).to(device=device)
            
            with amp.autocast(enabled=amp_enabled):
                tr_outputs = model(tr_inputs)
                tr_labelsloss = tr_labels.feats.squeeze(-1)
                tr_loss = criterion(tr_outputs.feats, tr_labelsloss)
            
            running_loss += tr_loss.item()
        
            optimizer.zero_grad()
            scaler.scale(tr_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        training_losses.append(running_loss / train_steps)
        print(f"[Epoch {epoch+1}] Running Loss: {running_loss / train_steps}")
    
        model.eval()
        torchsparse.backends.benchmark = True  # type: ignore
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, (batch_coords, batch_feats, batch_labels) in enumerate(val_loader):
                
                v_inputs_list = []
                v_labels_list = []
            
                for i in range(len(batch_coords)):
                    inputs_sparse = SparseTensor(coords=batch_coords[i], feats=batch_feats[i])
                    labels_sparse = SparseTensor(coords=batch_coords[i], feats=batch_labels[i])
                    v_inputs_list.append(inputs_sparse)
                    v_labels_list.append(labels_sparse)
            
                v_inputs = sparse_collate(v_inputs_list).to(device=device)
                v_labels = sparse_collate(v_labels_list).to(device=device)
        
                n_correct = 0
                
                with amp.autocast(enabled=True):
                    v_outputs = model(v_inputs)
                    v_labelsloss = v_labels.feats.squeeze(-1)
                    v_loss = criterion(v_outputs.feats, v_labelsloss)
                
                val_loss += v_loss.item()
                
        validation_losses.append(val_loss / val_steps)
        print(f"[Epoch {epoch+1}] Validation Loss: {val_loss / val_steps}")
    
    
    MODEL_PATH = f"../training/{datetime_str}/models/"
    LOSS_PATH = f"../training/{datetime_str}/loss_data/"
    
    if (not os.path.exists(MODEL_PATH)) and (not os.path.exists(LOSS_PATH)):
        os.makedirs(MODEL_PATH)
        os.makedirs(LOSS_PATH)
             
    filename = f"epochs{num_epochs}_lr{lr}_{datetime_str}.pth"
    torch.save(model.state_dict(), MODEL_PATH + filename)
    
    tr_filename = f"trainloss_{datetime_str}.npy"
    v_filename = f"valloss_{datetime_str}.npy"
    np.save(LOSS_PATH + tr_filename, training_losses)
    np.save(LOSS_PATH + v_filename, validation_losses)
    
    print('Finished Training')
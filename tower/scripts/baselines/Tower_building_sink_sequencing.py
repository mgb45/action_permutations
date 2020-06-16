#!/usr/bin/python

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("seed", type=int, help="Set seed here")
args = parser.parse_args()

import numpy as np
import glob

import torch
torch.manual_seed(args.seed)

import numpy as np
np.random.seed(args.seed)


import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.stats import kendalltau

from models import Sampler, SinkhornNet

# Load some data - restrict to first 150 demos
flist = sorted(glob.glob('../../demos/perms/order*'))


# Extract final images and object pick sequence
obj_list = []
im_list = []
for f in flist:
    run = int(f.split('_')[1][:-4])

    ims = np.load('../../demos/perms/ims_%04d.npy'%run)
    obj_ids = np.load('../../demos/perms/order_%04d.npy'%run)
    
    obj_list.append(obj_ids)
    im_list.append(ims)

# Create dataset 
dataset = Sampler(np.swapaxes(np.stack(im_list),1,3),np.stack(obj_list),6)

# Split into test/ train
train_dataset,test_dataset = torch.utils.data.random_split(dataset, [100,len(im_list)-100])

# Training data loader
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Build model
sn = SinkhornNet(latent_dim=16, image_channels=3, K=6)
optimizer = torch.optim.Adam(sn.parameters(), lr=3e-4)

n_epochs = 1000

# Train model
for j in range(n_epochs):
    
    for im, seq, seq_order in train_loader:
    
        loss, seq_pred = sn.loss(seq, im, seq_order)
            
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Evaluate model
sn.eval()
sn.noise_factor = 0.0
sn.n_samples = 1

# Compare pair ranks
tau_list = []
for im,seq,seq_ordered in test_loader:
    P = sn.predict_P(im)
    obj_ids = np.round(np.matmul(np.linalg.inv(P[0,:,:].detach().numpy()),seq_ordered[0,:].numpy())).astype(int)
    tau, _ = kendalltau(obj_ids, seq[0,:].numpy())
    tau_list.append(tau)

np.savetxt('../../exps/perms/tau_sink_%02d.txt'%args.seed,np.array(tau_list))

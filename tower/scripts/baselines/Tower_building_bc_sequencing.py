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
from scipy.optimize import linear_sum_assignment

from models import SamplerBC, BCNet

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

# Build dataset
dataset = SamplerBC(np.swapaxes(np.stack(im_list),1,3),np.stack(obj_list))

# Split data
train_dataset,test_dataset = torch.utils.data.random_split(dataset, [100,len(im_list)-100])

# Training data loader
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Build model
bc = BCNet(latent_dim=16, image_channels=3, K=6)
optimizer = torch.optim.Adam(bc.parameters(), lr=1e-4)

n_epochs = 1000

for j in range(n_epochs):
    
    for im, seq in train_loader:
    
        loss, seq_pred = bc.loss(seq, im)       
     
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
# Evaluate model
bc.eval()


# Compare ranks
tau_list = []
for im,seq in test_loader:
    seq_pred = bc(im)
    obj_ids = np.round(np.matmul(seq_pred[0,:,:].detach().numpy(),np.arange(6))).astype(int)
    #obj_ids = np.argmax(seq_pred[0,:,:].detach().numpy(),-1)
    tau, _ = kendalltau(obj_ids, seq[0,:].numpy())
    tau_list.append(tau)


# Constrain using Hungarian algorithm
tau_list_hung = []
for im,seq in test_loader:
    seq_pred = bc(im)
    _,obj_ids = linear_sum_assignment(1.0-seq_pred[0,:,:].detach().numpy())
    #obj_ids = np.argmax(seq_pred[0,:,:].detach().numpy(),-1)
    tau, _ = kendalltau(obj_ids, seq[0,:].numpy())
    tau_list_hung.append(tau)

np.savetxt('../../exps/perms/tau_bc_%02d.txt'%args.seed,np.array(tau_list))
np.savetxt('../../exps/perms/tau_bc_hung%02d.txt'%args.seed,np.array(tau_list_hung))


#!/usr/bin/python

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("demos", type=int, help="Set demonstration number here")
args = parser.parse_args()

import glob

import torch
torch.manual_seed(0)

import numpy as np
np.random.seed(0)
import random

random.seed(0)

import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.stats import kendalltau

# Set up pytorch dataloader
class Sampler(Dataset):
    
    def __init__(self, ims, actions,K=6):
        
        self.ims = torch.FloatTensor(ims.astype('float'))
        self.actions = torch.FloatTensor(actions.astype('float'))
        self.K = K
        
        
    def __len__(self):
        
        return self.ims.shape[0]
    
    def __getitem__(self, index):
        
        im = self.ims[index,:,:,:]
        actions = self.actions[index,:]
        return im, actions, torch.FloatTensor(np.arange(self.K).astype('float'))

# Define permuation prediction model
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class SinkhornNet(nn.Module):

    def __init__(self, latent_dim=16, image_channels=3, K=6, n_samples=5, noise_factor=1.0, temp=1.0, n_iters=20):
        super(SinkhornNet, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128, 256, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            Flatten(),
            nn.Linear(4096, latent_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        
        # Sinkhorn params
        self.latent_dim = latent_dim
        self.K = K
        self.n_samples = n_samples
        self.noise_factor = noise_factor
        self.temp = temp
        self.n_iters = n_iters
        
        self.criterion = nn.MSELoss()

        self.sinknet = nn.Sequential(
                        nn.Linear(self.latent_dim, K*K))
    
    def permute(self,seq,P):
        
        return torch.matmul(P,seq)
    
    def predict_P(self,im):
        
        latent = self.encoder(im)
        log_alpha = self.sinknet(latent)
        log_alpha = log_alpha.reshape(-1, self.K, self.K)
         
        soft_perms_inf, log_alpha_w_noise = self.gumbel_sinkhorn(log_alpha)
        
        P = self.inv_soft_pers_flattened(soft_perms_inf,self.K)
        return P
    
    def forward(self, seq, im):
        
        latent = self.encoder(im)
        log_alpha = self.sinknet(latent)
        log_alpha = log_alpha.reshape(-1, self.K, self.K)
         
        soft_perms_inf, log_alpha_w_noise = self.gumbel_sinkhorn(log_alpha)
        
        P = self.inv_soft_pers_flattened(soft_perms_inf,self.K)
        
        seq_tiled = seq.repeat(self.n_samples, 1)
        ordered  = self.permute(torch.unsqueeze(seq_tiled,dim=-1),P)
        
        return ordered
    
    def loss(self, seq, im, seq_gt):
        
        seq_pred = torch.squeeze(self.forward(seq,im))
        seq_gt = seq_gt.repeat(self.n_samples, 1)

        return self.criterion(seq_pred,seq_gt), seq_pred
    
    def inv_soft_pers_flattened(self,soft_perms_inf,n_numbers):
        inv_soft_perms = torch.transpose(soft_perms_inf, 2, 3)
        inv_soft_perms = torch.transpose(inv_soft_perms, 0, 1)

        inv_soft_perms_flat = inv_soft_perms.view(-1, n_numbers, n_numbers)
        return inv_soft_perms_flat
    
    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape).float()
        return -torch.log(eps - torch.log(U + eps))
    
    def gumbel_sinkhorn(self,log_alpha):
        
        n = log_alpha.size()[1]
        log_alpha = log_alpha.view(-1, n, n)
        batch_size = log_alpha.size()[0]

        log_alpha_w_noise = log_alpha.repeat(self.n_samples, 1, 1)

        if self.noise_factor == 0:
            noise = 0.0
        else:
            noise = self.sample_gumbel([self.n_samples*batch_size, n, n])*self.noise_factor

        log_alpha_w_noise = log_alpha_w_noise + noise
        log_alpha_w_noise = log_alpha_w_noise / self.temp

        my_log_alpha_w_noise = log_alpha_w_noise.clone()

        sink = self.sinkhorn(my_log_alpha_w_noise)

        sink = sink.view(self.n_samples, batch_size, n, n)
        sink = torch.transpose(sink, 1, 0)
        log_alpha_w_noise = log_alpha_w_noise.view(self.n_samples, batch_size, n, n)
        log_alpha_w_noise = torch.transpose(log_alpha_w_noise, 1, 0)

        return sink, log_alpha_w_noise
    
    def sinkhorn(self,log_alpha, n_iters = 20):
   
        n = log_alpha.size()[1]
        log_alpha = log_alpha.view(-1, n, n)

        for i in range(n_iters):
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True)).view(-1, n, 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True)).view(-1, 1, n)
        return torch.exp(log_alpha)


# Load some data - restrict to first N demos
flist = sorted(glob.glob('../../demos/perms_unique/order*'))
random.shuffle(flist)
train_list = flist[0:args.demos]
test_list = flist[args.demos:]


# Extract final images and object pick sequence
obj_list = []
im_list = []
for f in train_list:
    run = int(f.split('_')[2][:-4])

    ims = np.load('../../demos/perms_unique/ims_%04d.npy'%run)
    obj_ids = np.load('../../demos/perms_unique/order_%04d.npy'%run)
    
    obj_list.append(obj_ids)
    im_list.append(ims)

# Create dataset 
dataset = Sampler(np.swapaxes(np.stack(im_list),1,3),np.stack(obj_list),6)

# Training data loader
batch_size = 16
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Build model
sn = SinkhornNet(latent_dim=128, image_channels=3, K=6)
optimizer = torch.optim.Adam(sn.parameters(), lr=3e-4)

n_epochs = 1000

# Train model
for j in range(n_epochs):
    
    for im, seq, seq_order in train_loader:
    
        loss, seq_pred = sn.loss(seq, im, seq_order)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print("\r Epoch %d Loss: %2.2f"%(j,loss.item()),end=" ")

# Evaluate model
sn.eval()
sn.noise_factor = 0.0
sn.n_samples = 1

print('Evaluating...')
# Compare pair ranks
tau_list = []
acc_list = []
precision = []
for f in flist:
    run = int(f.split('_')[2][:-4])
    im = np.load('../../demos/perms_unique/ims_%04d.npy'%run)
    im = np.swapaxes(im.reshape(1,64,64,3),1,3)
    seq = np.load('../../demos/perms_unique/order_%04d.npy'%run)

    P = sn.predict_P(torch.from_numpy(im).float())
    obj_ids = np.round(np.matmul(np.linalg.inv(P[0,:,:].detach().numpy()),np.arange(6))).astype(int)
    tau, _ = kendalltau(obj_ids, seq)
    tau_list.append(tau)
    acc_list.append(np.array_equal(obj_ids,seq))
    precision.append(np.sum((obj_ids==seq))/(seq.shape[0]))

np.savetxt('../../exps/perms_unique/tau_sink_%04d.txt'%args.demos,np.array(tau_list))
np.savetxt('../../exps/perms_unique/acc_sink_%04d.txt'%args.demos,np.array(acc_list))
np.savetxt('../../exps/perms_subsets/precision_sink_%04d.txt'%args.demos,np.array(precision))

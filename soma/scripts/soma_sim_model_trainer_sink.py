#!/usr/bin/python

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("seed", type=int, help="Set seed number here")
args = parser.parse_args()

import numpy as np
np.random.seed(args.seed)
from matplotlib import pyplot as plt
from IPython import display
import torch
import torch.nn as nn
torch.manual_seed(args.seed)
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment
device = torch.device('cuda:0')

key = {'Orange':0,'Green':1,'Black':2,'Purple':3,'White':4,'LBlue':5,'Blue':6}


extraction_orders = np.genfromtxt('./data/extraction_order.txt',delimiter=',',dtype=str)

images = np.load('./data/cube_ims.npy')

actions = np.vectorize(key.get)(extraction_orders)

K = 7
a_one_hot = np.zeros((actions.shape[0],K,K))
for i,a in enumerate(actions):
    oh = np.zeros((K,K))
    oh[np.arange(a.shape[0]),a] = 1
    a_one_hot[i,:,:] = oh

class Sampler(Dataset):
    
    def __init__(self, ims, actions, K=6):
        
        self.ims = torch.FloatTensor(ims.astype('float'))
        self.actions = torch.FloatTensor(actions.astype('float'))
        self.indices = torch.FloatTensor(np.arange(ims.shape[0]))
        self.K = K
        
        
    def __len__(self):
        
        return self.ims.shape[0]
    
    def __getitem__(self, index):
        
        im = self.ims[index,:,:,:].reshape(-1,64,64)
        actions = self.actions[index,:,:]
        return im.to(device), actions.to(device), torch.eye(self.K).to(device),self.indices[index].to(device)

dataset = Sampler(np.swapaxes(np.stack(images),2,4),a_one_hot,7)

train_dataset,test_dataset = torch.utils.data.random_split(dataset, [180,60])

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class SinkhornNet(nn.Module):

    def __init__(self, latent_dim=16, image_channels=3, K=6, max_K=6, n_samples=5, noise_factor=1.0, temp=1.0, n_iters=5):
        super(SinkhornNet, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=5),
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
        self.max_K = max_K
        self.n_samples = n_samples
        self.noise_factor = noise_factor
        self.temp = temp
        self.n_iters = n_iters
        
        self.criterion = nn.MSELoss()

        self.sinknet = nn.Sequential(nn.Linear(self.latent_dim, self.latent_dim),
                                     nn.ReLU(),
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
    
    def loss(self, seq, im, seq_gt):
        
        seq_pred = self.forward(seq_gt,im)
        seq_pred = torch.squeeze(seq_pred)
        
        recon_loss = self.criterion(seq_pred[:,0:self.max_K,:],seq.repeat(self.n_samples, 1,1))
         
        return recon_loss, seq_pred
    
    def forward(self, seq, im):
        
        latent = self.encoder(im)
        log_alpha = self.sinknet(latent)
        log_alpha = log_alpha.reshape(-1, self.K, self.K)
         
        soft_perms_inf, log_alpha_w_noise = self.gumbel_sinkhorn(log_alpha)
        
        P = self.inv_soft_pers_flattened(soft_perms_inf,self.K)
        
        seq_tiled = seq.repeat(self.n_samples, 1, 1)
        ordered  = self.permute(seq_tiled,P)
        
        return ordered
    
    def loss(self, seq, im, seq_gt):
        
        seq_pred = self.forward(seq_gt,im)
        seq_pred = torch.squeeze(seq_pred)

        recon_loss = self.criterion(seq_pred,seq.repeat(self.n_samples, 1,1))
                
        return recon_loss, seq_pred
    
    def inv_soft_pers_flattened(self,soft_perms_inf,n_numbers):
        inv_soft_perms = torch.transpose(soft_perms_inf, 2, 3)
        inv_soft_perms = torch.transpose(inv_soft_perms, 0, 1)

        inv_soft_perms_flat = inv_soft_perms.view(-1, n_numbers, n_numbers)
        return inv_soft_perms_flat
    
    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape).float().to(device)
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
    
    def sinkhorn(self,log_alpha):
   
        n = log_alpha.size()[1]
        log_alpha = log_alpha.view(-1, n, n)

        for i in range(self.n_iters):
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True)).view(-1, n, 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True)).view(-1, 1, n)
        return torch.exp(log_alpha)

sn = SinkhornNet(latent_dim=1024, image_channels=12, K=7)
sn.to(device)
optimizer = torch.optim.Adam(sn.parameters(), lr=3e-4)
n_epochs = 5000
losses = []

plt.figure(figsize=(15,7))

for j in range(n_epochs):
    
    batch_losses = []
    for im, seq, seq_order,_ in train_loader:
    
        loss, seq_pred = sn.loss(seq, im, seq_order)
        
        batch_losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    losses.append(np.mean(batch_losses))

sn.eval()
sn.noise_factor = 0.0
sn.n_samples = 1
sn.n_iters = 100

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

prec_list = []
actions_pred = []
index_list = []
for im,seq,seq_ordered,indices in test_loader:
    P = sn.predict_P(im)
    
    index_list.append(indices.cpu().numpy())
    _,obj_ids = linear_sum_assignment(1-P[0,:,:].cpu().detach().numpy())
    actions_pred.append(obj_ids)
    prec = np.sum(obj_ids==np.argmax(seq.cpu().numpy(),-1))/7
    prec_list.append(prec)


np.mean(prec_list)


parts = [k[0] for k in key.items()]

pred_extractions = np.array(parts)[np.array(actions_pred).astype(int)]
indices = np.array(index_list).astype(int)

np.save('pred_order_%02d.npy'%args.seed,pred_extractions)
np.save('test_indices_%02d.npy'%args.seed,indices)


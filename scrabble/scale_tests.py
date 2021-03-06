import numpy as np
from matplotlib import pyplot as plt
from IPython import display
from scipy.stats import kendalltau

np.random.seed(0)

import random
random.seed(0)

import torch
torch.manual_seed(0)

import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
device = torch.device('cuda:1')

from model import Sampler, SinkhornNet

batch_size = 64
latent_dim = 128
learning_rate = 1e-4
n_epochs = 5000
max_actions = 98
min_actions = 6
action_step = 5
noise = 1.0
temp = 1.0
K = 6

for numActions in [26]:#list(range(min_actions,max_actions,action_step))[14:]:

    print('\nTesting for %d actions:'%numActions)
    train_actions = np.load('./data/actions_%02d.npy'%numActions,allow_pickle=True)
    train_ims = np.load('./data/word_pics_%02d.npy'%numActions)
 
    K = 6
    a_one_hot = np.zeros((train_actions.shape[0],numActions,numActions))
    seq_len = []
    for i,a in enumerate(train_actions):
        seq_len.append(a.shape[0]-1)
        oh = np.zeros((numActions,numActions))
        oh[np.arange(a.shape[0]),a] = 1
        a_one_hot[i,:,:] = oh

    # Create test train dataset
    dataset = Sampler(np.swapaxes(np.stack(train_ims),1,3),a_one_hot,np.array(seq_len),numActions)
    train_dataset,test_dataset = torch.utils.data.random_split(dataset,[10000,5000])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    sn = SinkhornNet(latent_dim=latent_dim, image_channels=3, max_K=6, n_samples=5, noise_factor=noise, temp=temp, n_iters=20, K=numActions)
    sn.to(device)
    optimizer = torch.optim.Adam(sn.parameters(), lr=learning_rate)

    for j in range(n_epochs):

        batch_losses = []
        batch_losses_m = []
        for im, seq, seq_order,seq_len in train_loader:

            loss_r,loss_m, seq_pred,mask = sn.loss(seq, im, seq_order,seq_len)

            batch_losses.append(loss_r.item())
            batch_losses_m.append(loss_m.item())

            loss = loss_r + loss_m
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        print('\rEpoch %d. Loss recon %4.2f, Loss mask  %2.2f.'%(j,np.mean(batch_losses),np.mean(batch_losses_m)),end='')


    # Evaluate model
    sn.eval()
    sn.noise_factor = 0.0
    sn.n_samples = 1

    tau_list = []
    acc = []
    precision = []
    obj_list = []
    spelling_precision = []
    
    action_set = np.load('./action_set.npy')
    
    for im,seq,seq_ordered,seq_len in test_loader:
        
        P = sn.predict_P(im)
        
        order,stop = sn(seq_ordered,im)
        
        l = (seq_len.cpu().numpy()[0]+1)
        stop_bin = np.argmax(stop.cpu().detach().numpy(),1)
        
        acc.append(stop_bin==(l-1))
        
        obj_ids = np.argmax(P[0,:,:].cpu().detach().numpy(),1)[:l]
        gt = np.argmax(seq[0,:l].cpu().numpy(),1)
        
        tau, _ = kendalltau(obj_ids,gt)
        tau_list.append(tau)

        precision.append(np.sum(obj_ids==gt)/l)
        spelling_precision.append(np.sum(action_set[obj_ids]==action_set[gt])/l)
        
    np.save('./data/obj_list_%02d.npy'%numActions,obj_list)

    np.save('./data/ntau_%02d.npy'%numActions,np.vstack(tau_list))
    np.save('./data/nacc_%02d.npy'%numActions,np.vstack(acc))
    np.save('./data/nprec_%02d.npy'%numActions,np.vstack(precision))
    np.save('./data/nsprec_%02d.npy'%numActions,np.vstack(spelling_precision))

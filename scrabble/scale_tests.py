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
device = torch.device('cuda:0')

from model import Sampler, SinkhornNet

batch_size = 32
latent_dim = 512
learning_rate = 3e-4
n_epochs = 1000
max_actions = 98
min_actions = 6
action_step = 5
noise = 1.0
temp = 1.5
K = 6

for numActions in list(range(min_actions,max_actions,action_step)):

    print('\nTesting for %d actions:'%numActions)
    train_actions = np.load('./data/actions_%02d.npy'%numActions,allow_pickle=True)
    train_ims = np.load('./data/word_pics_%02d.npy'%numActions)

    # Build masked input matrix
#     act_idxs = np.ones((len(train_actions),numActions))*100
#     seq_len = []
#     for j,act_idx in enumerate(train_actions):
#         seq_len.append(act_idx.shape[0]-1)
#         act_idxs[j,0:act_idx.shape[0]] = act_idx
        
    a_one_hot = np.zeros((actions.shape[0],K,numActions))
    for i,a in enumerate(actions):
        oh = np.zeros((actions.shape[1],K))
        oh[np.arange(actions.shape[1]),a] = 1
        a_one_hot[i,:,:] = oh

    # Create test train dataset
    dataset = Sampler(np.swapaxes(np.stack(train_ims),1,3),a_one_hot,np.array(seq_len),numActions)
    train_dataset,test_dataset = torch.utils.data.random_split(dataset,[10000,5000])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    sn = SinkhornNet(latent_dim=latent_dim, image_channels=3, max_K=6, n_samples=5, noise_factor=noise, temp=temp, n_iters=20, K=numActions)
    sn.to(device)
    optimizer = torch.optim.Adam(sn.parameters(), lr=learning_rate)

    pre_loss = 0
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
        delta_loss = np.abs((np.mean(batch_losses)+np.mean(batch_losses_m)) - pre_loss)
        pre_loss = (np.mean(batch_losses)+np.mean(batch_losses_m))
        if delta_loss < 0.001:
            break;

    # Evaluate model
    sn.eval()
    sn.noise_factor = 0.0
    sn.n_samples = 1

    tau_list = []
    Acc = []
    for im,seq,seq_ordered,seq_len in test_loader:
        P = sn.predict_P(im)
        order,stop = sn(seq_ordered,im)
        Acc.append(np.argmax(stop.cpu().detach().numpy(),1)==(seq_len.cpu().numpy()))
        obj_ids = np.argmax(P[0,:,:].cpu().detach().numpy(),1)
        tau, _ = kendalltau(obj_ids[0:(seq_len.cpu().numpy()[0]+1)], seq[0,0:(seq_len.cpu().numpy()[0]+1)].cpu().numpy())
        tau_list.append(tau)

    np.save('./data/ntau_%02d.npy'%numActions,np.vstack(tau_list))
    np.save('./data/nacc_%02d.npy'%numActions,np.vstack(Acc))

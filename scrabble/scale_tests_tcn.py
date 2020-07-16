import numpy as np
from matplotlib import pyplot as plt
from IPython import display

np.random.seed(0)

import random
random.seed(0)

import torch
torch.manual_seed(0)

import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils import weight_norm
device = torch.device('cuda:1')

from model import SamplerTCN, TCNNet

batch_size = 64
latent_dim = 16
learning_rate = 1e-4
n_epochs = 5000
max_actions = 98
min_actions = 6
action_step = 5

from scipy.optimize import linear_sum_assignment

for numActions in [86]:#list(range(min_actions,max_actions,action_step))[3:]:

    print('\nTesting for %d actions:'%numActions)
    train_actions = np.load('./data/actions_%02d.npy'%numActions,allow_pickle=True)
    train_ims = np.load('./data/word_pics_%02d.npy'%numActions)
 
    obj_idxs = np.ones((train_actions.shape[0],numActions+1))*numActions
    seq_len = []
    for i,a in enumerate(train_actions):
        seq_len.append(a.shape[0])
        obj_idxs[i,0:a.shape[0]] = a

    # Create test train dataset
    dataset = SamplerTCN(np.swapaxes(np.stack(train_ims),1,3),obj_idxs,np.array(seq_len))
    train_dataset,test_dataset = torch.utils.data.random_split(dataset,[10000,5000])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    bc = TCNNet(latent_dim=latent_dim, image_channels=3, K=numActions+1,l=7)
    bc.to(device)
    optimizer = torch.optim.Adam(bc.parameters(), lr=learning_rate)

    for j in range(n_epochs):

        batch_losses = []
        for im, seq, seq_len in train_loader:

            loss, seq_pred = bc.loss(seq, im)
            batch_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        print('\rEpoch %d. Loss %4.2f,'%(j,np.mean(batch_losses)),end='')

    # Evaluate model
    bc.eval()
    
    acc = []
    precision = []
    obj_list = []
    spelling_precision = []
    
    action_set = np.load('./action_set.npy')
    
    for im,seq,seq_len in test_loader:
        
        logits = bc(im)
        
        l = seq_len.cpu().numpy()[0]
        _, obj_ids = linear_sum_assignment(1-logits[0,0:l,:].cpu().detach().numpy())
        stop = np.where(np.argmax(logits[0,:,:].cpu().detach().numpy(),1)==numActions)[0][0]-1
        
        acc.append(stop==(l-1))
       
        gt = seq[0,:l].cpu().numpy()
      
        precision.append(np.sum(obj_ids==gt)/l)
        spelling_precision.append(np.sum(action_set[obj_ids]==action_set[gt])/l)
        
    np.save('./data/obj_list_%02d.npy'%numActions,obj_list)

    np.save('./data/tcn_acc_%02d.npy'%numActions,np.vstack(acc))
    np.save('./data/tcn_prec_%02d.npy'%numActions,np.vstack(precision))
    np.save('./data/tcn_sprec_%02d.npy'%numActions,np.vstack(spelling_precision))

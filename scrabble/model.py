import torch
torch.manual_seed(0)

import numpy as np
np.random.seed(0)

import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
device = torch.device('cuda:0')
# Set up pytorch dataloader
class Sampler(Dataset):
    
    def __init__(self, ims, actions, seq_lens, K=6):
        
        self.ims = torch.FloatTensor(ims.astype('float')).to(device)
        self.actions = torch.FloatTensor(actions.astype('float')).to(device)
        self.seq_lens = torch.LongTensor(seq_lens.astype('int')).to(device)
        self.K = K
        
        
    def __len__(self):
        
        return self.ims.shape[0]
    
    def __getitem__(self, index):
        
        im = self.ims[index,:,:,:]/255.0
        actions = self.actions[index,:,:]
        seq_len = self.seq_lens[index]
        return im, actions, torch.eye(self.K).to(device), seq_len

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class SinkhornNet(nn.Module):

    def __init__(self, latent_dim=16, image_channels=3, K=6, max_K=6, n_samples=5, noise_factor=1.0, temp=1.0, n_iters=5):
        super(SinkhornNet, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3,3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128, 256, kernel_size=(3,3)),
            nn.ReLU(),
            Flatten(),
            nn.Linear(6656, latent_dim),
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
        
        self.criterion = nn.MSELoss(reduction='none')
        self.mask_criterion = nn.CrossEntropyLoss()

        self.sinknet = nn.Sequential(nn.Linear(self.latent_dim, self.latent_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.latent_dim, self.latent_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.latent_dim, K*K))
        
        self.masknet = nn.Sequential(
                        nn.Linear(self.latent_dim, self.latent_dim),
                        nn.ReLU(),
                        nn.Linear(self.latent_dim, self.max_K),nn.Softmax(dim=1))
    
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

        stopping = self.masknet(latent)
        
        return ordered, stopping
    
    def loss(self, seq, im, seq_gt, seq_len):
        
        seq_pred,stopping_bin = self.forward(seq_gt,im)
        seq_pred = torch.squeeze(seq_pred)

        mask_loss = self.mask_criterion(stopping_bin,seq_len)
        
        mask_idxs = torch.arange(0,self.K).reshape(1,-1,1).repeat(seq_pred.size(0),1,self.K).to(device)
        
        seq_len_mask = seq_len.reshape(-1,1,1).repeat(self.n_samples,self.K,self.K)
        
       
       
        stopping_mask = (mask_idxs<seq_len_mask).int().float()
       
        recon_loss = self.criterion(seq_pred*stopping_mask,seq.repeat(self.n_samples, 1,1)*stopping_mask)
        
        recon_loss = (recon_loss.sum(dim=-1).sum(dim=-1)/seq_len.repeat(self.n_samples).int().float()).mean()
        
        return recon_loss, mask_loss, seq_pred,stopping_mask
    
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

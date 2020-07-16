import torch
torch.manual_seed(0)

import numpy as np
np.random.seed(0)

import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.nn.utils import weight_norm

device = torch.device('cuda:1')


class SamplerTCN(Dataset):
    
    def __init__(self, ims, actions, seq_lens):
        
        self.ims = torch.FloatTensor(ims.astype('float')).to(device)
        self.actions = torch.LongTensor(actions.astype('int')).to(device)
        self.seq_lens = torch.LongTensor(seq_lens.astype('int')).to(device)
        
        
    def __len__(self):
        
        return self.ims.shape[0]
    
    def __getitem__(self, index):
        
        im = self.ims[index,:,:,:]
        actions = self.actions[index,:]
        seq_len = self.seq_lens[index]
        return im, actions, seq_len
    

# Set up pytorch dataloader
class Sampler(Dataset):
    
    def __init__(self, ims, actions, seq_lens, K=6):
        
        self.ims = torch.FloatTensor(ims.astype('float'))
        self.actions = torch.FloatTensor(actions.astype('float'))
        self.seq_lens = torch.LongTensor(seq_lens.astype('int'))
        self.K = K
        
        
    def __len__(self):
        
        return self.ims.shape[0]
    
    def __getitem__(self, index):
        
        im = self.ims[index,:,:,:]/255.0 -0.5
        actions = self.actions[index,:,:]
        seq_len = self.seq_lens[index]
        return im.to(device), actions.to(device), torch.eye(self.K).to(device), seq_len.to(device)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class SinkhornNet(nn.Module):

    def __init__(self, latent_dim=16, image_channels=3, K=6, max_K=6, n_samples=5, noise_factor=1.0, temp=1.0, n_iters=5):
        super(SinkhornNet, self).__init__()
        
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=(3,3)),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=(3,3)),
#             nn.ReLU(),
#             nn.MaxPool2d(2,2),
#             nn.Conv2d(64, 128, kernel_size=(3,3)),
#             nn.ReLU(),
#             nn.MaxPool2d(2,2),
#             nn.Conv2d(128, 256, kernel_size=(3,3)),
#             nn.ReLU(),
#             Flatten(),
#             nn.Linear(6656, latent_dim),
#             nn.ReLU(),
#             nn.Dropout(p=0.5)
#         )



        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(512, latent_dim)
        self.encoder = model
        
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

        self.sinknet = nn.Sequential(nn.Linear(self.latent_dim, K*K))
        
        self.masknet = nn.Sequential(nn.Linear(self.latent_dim, self.max_K),nn.Softmax(dim=1))
    
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
        
       
       
        stopping_mask = (mask_idxs<=seq_len_mask).int().float()
       
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
    

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class TCNNet(nn.Module):

    def __init__(self, latent_dim=16, image_channels=3, K=6,l=6):
        super(TCNNet, self).__init__()
        
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(512, latent_dim*l)
        self.encoder = model
        
        self.tcn = TemporalConvNet(l,[K]*6)
        
        self.latent_dim = latent_dim
        self.K = K
        self.l = l
        
        self.criterion = nn.CrossEntropyLoss()

        self.fc = nn.Sequential(
                         nn.Linear(latent_dim, K))
    
    def forward(self, im):
        
        latent = self.encoder(im)
        
        stacked_latent = torch.reshape(latent,(-1,self.l,self.latent_dim))
        
        tc = self.tcn(stacked_latent)
        
        y = self.fc(tc)
        logits = torch.nn.functional.softmax(y,dim=-1)
        
        
        return logits
    
    def loss(self, seq, im):
        
        seq_logits = self.forward(im)
        
        recon = self.criterion(seq_logits.view(-1,self.K),seq.view(-1))

        return recon, seq_logits


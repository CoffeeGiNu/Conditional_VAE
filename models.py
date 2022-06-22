import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, x_dim, z_dim, num_classes, device='cpu') -> None:
        super(Encoder, self).__init__()
        self.eps = np.spacing(1)
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.device = device
        self.num_classes = num_classes
        
        self.fc1 = nn.Linear(x_dim+self.num_classes, 512)
        self.fc2 = nn.Linear(512, 256)
        self.mean = nn.Linear(256, z_dim)
        self.log_var = nn.Linear(256, z_dim)
    
    def forward(self, x, l):
        l_onehot = F.one_hot(l, num_classes=self.num_classes).to(self.device)
        x_concat = torch.cat((x, l_onehot), dim=1)

        x = torch.relu(self.fc1(x_concat))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        log_var = self.log_var(x)

        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, x_dim, z_dim, num_classes, device='cpu') -> None:
        super(Decoder, self).__init__()
        self.eps = np.spacing(1)
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.device = device
        self.num_classes = num_classes
        
        self.fc1 = nn.Linear(z_dim+self.num_classes, 256)
        self.fc2 = nn.Linear(256, 512)
        self.drop = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(512, 784)
    
    def forward(self, z, l):
        l_onehot = F.one_hot(l, num_classes=self.num_classes).to(self.device)
        z_concat = torch.cat((z, l_onehot), dim=1)

        z = torch.relu(self.fc1(z_concat))
        z = torch.relu(self.fc2(z))
        z = self.drop(z)
        dec_y = torch.sigmoid(self.fc3(z))

        return dec_y


class ConditionalVariationalAutoEncoder(nn.Module):
    def __init__(self, x_dim, z_dim, num_classes, device) -> None:
        super(ConditionalVariationalAutoEncoder, self).__init__()
        self.eps = np.spacing(1)
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.device = device
        self.encoder = Encoder(x_dim, z_dim, num_classes, device)
        self.decoder = Decoder(x_dim, z_dim, num_classes, device)
    
    def pseudo_sample(self, mean, log_var):
        rand = torch.randn(mean.shape, device=self.device)
        z = mean + rand * torch.exp(1/2 * log_var)

        return z
    
    def forward(self, x, l):
        x = x.view(-1, self.x_dim)
        mean, log_var = self.encoder(x, l)
        z = self.pseudo_sample(mean, log_var)
        y = self.decoder(z, l)
        KLD = 1/2 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var))
        rc = torch.sum(x * torch.log(y + self.eps) 
           + (1 - x) * torch.log(1 - y + self.eps))

        return [KLD, rc], z, y

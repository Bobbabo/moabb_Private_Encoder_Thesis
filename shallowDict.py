from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class ShallowPrivateCollapsedDictNetSlow(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times=1001, dropout=0.5, num_kernels=40, kernel_size=25, pool_size=100, num_subjects=9):
        super(ShallowPrivateCollapsedDictNetSlow, self).__init__()
        self.num_subjects = num_subjects
        
        self.spatio_temporal_layers = nn.ModuleDict({
            f'subject_{i+1}': nn.Conv2d(n_chans, num_kernels, (1, kernel_size))
            for i in range(num_subjects)
        })   

        self.pool = nn.AvgPool2d((1, pool_size))
        self.batch_norm = nn.BatchNorm2d(num_kernels)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.LazyLinear(n_outputs)

    def forward(self, x):
           
        subject_ids = x[:, 0, -1] / 1000000  # Assuming subject IDs are in the last time point of channel 0
        x = x[:, :, :-1]  # Remove the last time point (which contains the subject ID)
        
        x = torch.unsqueeze(x, dim=2)
        outputs = []
        for i in range(x.size(0)):
            subject_id = subject_ids[i].long().item()
            x_i = x[i:i+1]
            x_i = self.spatio_temporal_layers[f'subject_{subject_id}'](x_i)
            outputs.append(x_i)
        x = torch.cat(outputs, dim=0)
        x = F.elu(x)
        x = self.batch_norm(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    

class ShallowPrivateSpatialDictNetSlow(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times=1001, dropout=0.5, num_kernels=40, kernel_size=25, pool_size=100, num_subjects=9):
        super(ShallowPrivateSpatialDictNetSlow, self).__init__()
        self.num_subjects = num_subjects
        
        self.temporal = nn.Conv2d(1, num_kernels, (1, kernel_size))
        
        self.spatial_layers = nn.ModuleDict({
            f'subject_{i+1}': nn.Conv2d(num_kernels, num_kernels, (n_chans, 1))
            for i in range(num_subjects)
        })        
        
        self.pool = nn.AvgPool2d((1, pool_size))
        self.batch_norm = nn.BatchNorm2d(num_kernels)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.LazyLinear(n_outputs)

    def forward(self, x):
        
        subject_ids = x[:, 0, -1] / 1000000  # Assuming subject IDs are in the last time point of channel 0
        x = x[:, :, :-1]  # Remove the last time point (which contains the subject ID)
        
        x = torch.unsqueeze(x, dim=1)
        x = self.temporal(x)
        
        outputs = []
        for i in range(x.size(0)):
            subject_id = subject_ids[i].long().item()
            x_i = x[i:i+1]
            x_i = self.spatial_layers[f'subject_{subject_id}'](x_i)
            outputs.append(x_i)
        x = torch.cat(outputs, dim=0)

        x = F.elu(x)
        x = self.batch_norm(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class ShallowPrivateTemporalDictNetSlow(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times=1001, dropout=0.5, num_kernels=40, kernel_size=25, pool_size=100, num_subjects=9):
        super(ShallowPrivateTemporalDictNetSlow, self).__init__()
        self.num_subjects = num_subjects
        
        self.temporal_layers = nn.ModuleDict({
            f'subject_{i+1}': nn.Conv2d(1, num_kernels, (1, kernel_size))
            for i in range(num_subjects)
        })
        
        self.spatial = nn.Conv2d(num_kernels, num_kernels, (n_chans, 1))
        self.pool = nn.AvgPool2d((1, pool_size))
        self.batch_norm = nn.BatchNorm2d(num_kernels)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.LazyLinear(n_outputs)

    def forward(self, x):
        
        subject_ids = x[:, 0, -1] / 1000000  # Assuming subject IDs are in the last time point of channel 0
        x = x[:, :, :-1]  # Remove the last time point (which contains the subject ID)
        
        x = torch.unsqueeze(x, dim=1)
        outputs = []
        for i in range(x.size(0)):
            subject_id = subject_ids[i].long().item()
            x_i = x[i:i+1]
            x_i = self.temporal_layers[f'subject_{subject_id}'](x_i)
            outputs.append(x_i)
        x = torch.cat(outputs, dim=0)
        x = self.spatial(x)
        x = F.elu(x)
        x = self.batch_norm(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
    
class SubjectDicionaryFCNet(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times=1001, dropout=0.5, num_kernels=40, 
                kernel_size=25, pool_size=100, num_subjects=9):
        super(SubjectDicionaryFCNet, self).__init__()
        self.num_subjects = num_subjects
        self.spatio_temporal = nn.Conv2d(
            n_chans, num_kernels, (1, kernel_size))
        self.pool = nn.AvgPool2d((1, pool_size))
        self.batch_norm = nn.BatchNorm2d(num_kernels)
        self.dropout = nn.Dropout(dropout)
        
        # Create a separate fully connected layer for each subject
        self.fc_layers = nn.ModuleDict({
            f'subject_{i+1}': nn.Linear(num_kernels * ((n_times - kernel_size + 1) // pool_size), n_outputs)
            for i in range(num_subjects)           
        })


    def forward(self, x):
        # Extract subject IDs from the last time point of the first channel (or any specific channel)
        subject_ids = x[:, 0, -1]/1000000   # Assuming subject IDs are in the last time point of channel 0
        x = x[:, :, :-1]  # Remove the last time point (which contains the subject ID)

        # Continue with the rest of the network
        x = torch.unsqueeze(x, dim=2)  # Add dimension for Conv2d
        x = self.spatio_temporal(x)
        x = F.elu(x)
        x = self.batch_norm(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)

        out = torch.zeros(x.size(0), self.fc_layers['subject_1'].out_features, device=x.device)

        # Use the subject IDs to select the appropriate FC layer
        for i in range(x.size(0)):  # Loop over batch size
            subject_id = subject_ids[i].item()  # Get the subject ID for the i-th sample
            fc_layer = self.fc_layers[f'subject_{int(subject_id)}']  # Select the appropriate FC layer
            out[i] = fc_layer(x[i])  # Apply FC layer to the i-th sample

        return out
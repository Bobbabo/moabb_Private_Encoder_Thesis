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
        self.instance_norm = nn.InstanceNorm2d(num_kernels) # Reflect on the use of InstanceNorm2d
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.LazyLinear(n_outputs)

    def forward(self, x):
           
        subject_ids = x[:, 0, -1] / 1000000  # Assuming subject IDs are in the last time point of channel 0
        x = x[:, :, :-1]  # Remove the last time point (which contains the subject ID)
        
        # make a list of unique subject IDs
        unique_subject_ids = torch.unique(subject_ids)
        
        if(unique_subject_ids.size(0) != 1):
            print("Error: More than one subject ID detected in the batch")
            return None
        
        subject_id = subject_ids[0].long().item()
        
        x = torch.unsqueeze(x, dim=2)
        
        x = self.spatio_temporal_layers[f'subject_{subject_id}'](x)
        
        x = F.elu(x)
        x = self.instance_norm(x)
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
        
        unique_subject_ids = torch.unique(subject_ids)
        
        if(unique_subject_ids.size(0) != 1):
            print("Error: More than one subject ID detected in the batch")
            return None
        
        x = torch.unsqueeze(x, dim=1)
        x = self.temporal(x)
        
        subject_id = subject_ids[0].long().item()
        
        x = self.spatial_layers[f'subject_{subject_id}'](x)

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
        
        # Given groups=1, weight of size [40, 1, 1, 25], expected input[1, 128, 22, 1000] to have 1 channels, but got 128 channels instead
        
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
        subject_id = subject_ids[0].long().item()
        
        unique_subject_ids = torch.unique(subject_ids)
        
        # if(unique_subject_ids.size(0) != 1):
        #     print("Error: More than one subject ID detected in the batch")
        #     return None
        x = torch.unsqueeze(x, dim=1)
        x = self.temporal_layers[f'subject_{subject_id}'](x)
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


        unique_subject_ids = torch.unique(subject_ids)
        subject_id = subject_ids[0].long().item()
        
        if(unique_subject_ids.size(0) != 1):
            print("Error: More than one subject ID detected in the batch")
            return None
        
        
        # Continue with the rest of the network
        x = torch.unsqueeze(x, dim=2)  # Add dimension for Conv2d
        x = self.spatio_temporal(x)
        x = F.elu(x)
        x = self.batch_norm(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        
        x = self.fc_layers[f'subject_{subject_id}'](x)

        return x
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

class SubjectDicionaryConvNet(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times=1001, dropout=0.5, num_kernels=40, 
                 kernel_size=25, pool_size=100, num_subjects=9):
        super(SubjectDicionaryConvNet, self).__init__()
        self.num_subjects = num_subjects

        # Create a dictionary of spatio-temporal convolutional layers, one per subject
        self.spatio_temporal_layers = nn.ModuleDict({
            f'subject_{i+1}': nn.Conv2d(n_chans, num_kernels, (1, kernel_size))
            for i in range(num_subjects)
        })

        self.pool = nn.AvgPool2d((1, pool_size))
        self.batch_norm = nn.BatchNorm2d(num_kernels)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.LazyLinear(n_outputs)

    def forward(self, x):
        # Extract subject IDs from the last time point of the first channel
        subject_ids = x[:, 0, -1] / 1000000  # Assuming subject IDs are stored scaled in the last time point
        x = x[:, :, :-1]  # Remove the last time point (which contains the subject ID)

        # Add dimension for Conv2d
        x = torch.unsqueeze(x, dim=2)  # Shape: (batch_size, n_chans, 1, n_times)

        # Prepare a list to collect outputs
        conv_outputs = []

        for i in range(x.size(0)):  # Loop over batch size
            subject_id = int(subject_ids[i].item())
            # Select the appropriate convolutional layer
            conv_layer = self.spatio_temporal_layers[f'subject_{subject_id}']
            # Apply the convolutional layer to the i-th sample
            xi = x[i].unsqueeze(0)  # Add batch dimension
            xi = conv_layer(xi)
            conv_outputs.append(xi)

        # Stack the outputs along the batch dimension
        x = torch.cat(conv_outputs, dim=0)

        x = F.elu(x)
        x = self.batch_norm(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = self.fc(x)
        return x



class SubjectOneHotConvNet2(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times=1001, dropout=0.5,
                 num_kernels=40, kernel_size=25, pool_size=100, num_subjects=9):
        super(SubjectOneHotConvNet2, self).__init__()
        self.num_subjects = num_subjects
        self.n_outputs = n_outputs
        self.num_kernels = num_kernels

        # Spatio-temporal convolutional layer with output channels partitioned per subject
        self.spatio_temporal = nn.Conv2d(
            n_chans, num_kernels * num_subjects, (1, kernel_size))

        self.pool = nn.AvgPool2d((1, pool_size))
        self.batch_norm = nn.BatchNorm2d(num_kernels)
        self.dropout = nn.Dropout(dropout)

        # Calculate the size of the output after convolution and pooling
        conv_output_size = ((n_times - kernel_size + 1) - (pool_size - 1) - 1) // pool_size + 1
        fc_input_size = num_kernels * conv_output_size
        self.fc = nn.Linear(fc_input_size, n_outputs)

    def forward(self, x):
        # Extract subject IDs from the last time point of the first channel
        subject_ids = x[:, 0, -1] / 1000000  # Adjust as per your data format
        x = x[:, :, :-1]  # Remove the last time point containing subject IDs

        # One-hot encode the subject IDs
        subject_one_hot = F.one_hot(subject_ids.long() - 1, num_classes=self.num_subjects).float()

        x = torch.unsqueeze(x, dim=2)  # Add dimension for Conv2d
        x = self.spatio_temporal(x)  # Shape: (batch_size, num_kernels * num_subjects, 1, conv_output_size)

        # Reshape to separate outputs per subject
        batch_size, total_kernels, _, conv_output_size = x.size()
        x = x.view(batch_size, self.num_subjects, self.num_kernels, 1, conv_output_size)

        # Use one-hot encoding to select outputs corresponding to the current subject
        subject_one_hot = subject_one_hot.view(batch_size, self.num_subjects, 1, 1, 1)
        x = x * subject_one_hot  # Element-wise multiplication
        x = x.sum(dim=1)  # Sum over the subject dimension

        x = F.elu(x)
        x = self.batch_norm(x)
        x = self.pool(x)
        x = x.view(batch_size, -1)  # Flatten
        x = self.dropout(x)
        x = self.fc(x)
        return x

class SubjectOneHotConvNet(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times=1001, dropout=0.5, num_kernels=40, 
                 kernel_size=25, pool_size=100, num_subjects=9):
        super(SubjectOneHotConvNet, self).__init__()
        self.num_subjects = num_subjects
        self.num_kernels = num_kernels
        self.spatio_temporal = nn.Conv2d(
            n_chans, num_kernels * num_subjects, (1, kernel_size))
        self.pool = nn.AvgPool2d((1, pool_size))
        self.batch_norm = nn.BatchNorm2d(num_kernels)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.LazyLinear(n_outputs)

    def forward(self, x):
        # Extract subject IDs from the last time point of the first channel
        subject_ids = x[:, 0, -1] / 1000000  # Assuming subject IDs are in the last time point of channel 0
        x = x[:, :, :-1]  # Remove the last time point (which contains the subject ID)

        # One-hot encode the subject IDs
        subject_one_hot = F.one_hot(subject_ids.long() - 1, num_classes=self.num_subjects).float()
        subject_one_hot = subject_one_hot.view(x.size(0), self.num_subjects, 1, 1, 1)
        
        # Continue with the rest of the network
        x = torch.unsqueeze(x, dim=2)  # Shape: (batch_size, n_chans, 1, n_times)
        x = self.spatio_temporal(x)     # Shape: (batch_size, num_kernels * num_subjects, 1, output_time_length)

        # Reshape x to separate the subject dimension
        x = x.view(x.size(0), self.num_subjects, self.num_kernels, 1, x.size(-1))

        # Multiply x by the subject one-hot encoding to select the correct subject's kernels
        x = x * subject_one_hot

        # Sum over the subject dimension to get the final output
        x = x.sum(dim=1)  # Shape: (batch_size, num_kernels, 1, output_time_length)

        x = F.elu(x)
        x = self.batch_norm(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class SubjectOneHotNet(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times=1001, dropout=0.5, num_kernels=40, 
                 kernel_size=25, pool_size=100, num_subjects=9):
        super(SubjectOneHotNet, self).__init__()
        self.n_outputs = n_outputs
        self.num_subjects = num_subjects
        self.spatio_temporal = nn.Conv2d(n_chans, num_kernels, (1, kernel_size))
        self.pool = nn.AvgPool2d((1, pool_size))
        self.batch_norm = nn.BatchNorm2d(num_kernels)
        self.dropout = nn.Dropout(dropout)
        
        # Shared fully connected layer with input based on kernel output
        self.fc_shared = nn.Linear(num_kernels * ((n_times - kernel_size + 1) // pool_size), n_outputs * num_subjects)

    def forward(self, x):
        # Extract subject IDs from the last time point of the first channel (or any specific channel)
        subject_ids = x[:, 0, -1] / 1000000  # Assuming subject IDs are in the last time point of channel 0
        x = x[:, :, :-1]  # Remove the last time point (which contains the subject ID)

        # One-hot encode the subject IDs
        subject_one_hot = F.one_hot(subject_ids.long() - 1, num_classes=self.num_subjects).float()
        
        # Continue with the rest of the network
        x = torch.unsqueeze(x, dim=2)  # Add dimension for Conv2d
        x = self.spatio_temporal(x)
        x = F.elu(x)
        x = self.batch_norm(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        
        # Apply the shared fully connected layer
        fc_output = self.fc_shared(x)  # Output shape: (batch_size, n_outputs * num_subjects)
        
        # Reshape output to separate each subject's output
        fc_output = fc_output.view(-1, self.num_subjects, self.n_outputs)
        
        # Multiply by the one-hot encoding to select the correct subject's output
        out = torch.einsum('bi,bio->bo', subject_one_hot, fc_output)

        return out

class SubjectAdvIndexFCNet(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times=1001, dropout=0.5, num_kernels=40, 
                 kernel_size=25, pool_size=100, num_subjects=9):
        super(SubjectAdvIndexFCNet, self).__init__()
        self.n_outputs = n_outputs
        self.num_subjects = num_subjects
        self.spatio_temporal = nn.Conv2d(n_chans, num_kernels, (1, kernel_size))
        self.pool = nn.AvgPool2d((1, pool_size))
        self.batch_norm = nn.BatchNorm2d(num_kernels)
        self.dropout = nn.Dropout(dropout)
        
        # Shared fully connected layer with expanded output
        fc_input_features = num_kernels * ((n_times - kernel_size + 1) // pool_size)
        fc_output_features = n_outputs * num_subjects
        self.fc_shared = nn.Linear(fc_input_features, fc_output_features)

    def forward(self, x):
        # Extract subject IDs from the last time point of the first channel
        subject_ids = x[:, 0, -1] / 1000000  # Assuming subject IDs are in the last time point of channel 0
        x = x[:, :, :-1]  # Remove the last time point (which contains the subject ID)
        subject_ids = (subject_ids.long() - 1)  # Convert to zero-based index
        
        # Continue with the rest of the network
        x = torch.unsqueeze(x, dim=2)  # Add dimension for Conv2d
        x = self.spatio_temporal(x)
        x = F.elu(x)
        x = self.batch_norm(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        
        # Apply the shared fully connected layer
        fc_output = self.fc_shared(x)  # Output shape: (batch_size, n_outputs * num_subjects)
        fc_output = fc_output.view(-1, self.num_subjects, self.n_outputs)
        
        
        # Use advanced indexing to select the correct subject's output
        batch_indices = torch.arange(fc_output.size(0), device=x.device)
        out = fc_output[batch_indices, subject_ids, :]  # Shape: (batch_size, n_outputs)
        
        return out



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
        


class CollapsedShallowNet(nn.Module):
    """
    A version of the ShallowFBCSPNet model with a combined spatiotemporal convolution instead of separate temporal and spatial convolutions

    Args:
        n_chans (int): Number of input channels.
        n_outputs (int): Number of output classes.
        n_times (int, optional): Number of timepoints in the input.
        dropout (float, optional): Dropout probability. Defaults to 0.5.
        num_kernels (int, optional): Number of kernels in the spatiotemporal convolution. Defaults to 40.
        kernel_size (int, optional): Size of the kernel in the spatiotemporal convolution. Defaults to 25.
        pool_size (int, optional): Size of the pooling window in the spatiotemporal convolution. Default is 100.
    """
    def __init__(self, n_chans, n_outputs, n_times=1001, dropout=0.5, num_kernels=40, kernel_size=25, pool_size=100):
        super(CollapsedShallowNet, self).__init__()
        self.spatio_temporal = nn.Conv2d(
            n_chans, num_kernels, (1, kernel_size))
        self.pool = nn.AvgPool2d((1, pool_size))
        self.batch_norm = nn.BatchNorm2d(num_kernels)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.LazyLinear(n_outputs)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=2)
        x = self.spatio_temporal(x)
        x = F.elu(x)
        x = self.batch_norm(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class ShallowFBCSPNet(nn.Module):
    """An implementation of the ShallowFBCSPNet model from https://arxiv.org/abs/1703.05051 

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        timepoints (int, optional): Number of timepoints in the input data. Default is 1001.
        dropout (float, optional): Dropout probability. Default is 0.5.
        num_kernels (int, optional): Number of convolutional kernels. Default is 40.
        kernel_size (int, optional): Size of the convolutional kernels. Default is 25.
        pool_size (int, optional): Size of the pooling window. Default is 100.
    """

    def __init__(self, n_chans, n_outputs, n_times=1001, dropout=0.5, num_kernels=40, kernel_size=25, pool_size=100):
        super(ShallowFBCSPNet, self).__init__()
        self.temporal = nn.Conv2d(1, num_kernels, (1, kernel_size))
        self.spatial = nn.Conv2d(num_kernels, num_kernels, (n_chans, 1))
        self.pool = nn.AvgPool2d((1, pool_size))
        self.batch_norm = nn.BatchNorm2d(num_kernels)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.LazyLinear(n_outputs)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = self.temporal(x)
        x = self.spatial(x)
        x = F.elu(x)
        x = self.batch_norm(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    

class CollapsedConformer(nn.Module):
    """
    A version of the Conformer model with a combined spatiotemporal convolution instead of separate temporal and spatial convolutions.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        timepoints (int, optional): Number of timepoints in the input data. Default is 1001.
        dropout (float, optional): Dropout rate. Default is 0.5.
        num_kernels (int, optional): Number of kernels in the spatiotemporal convolution. Default is 40.
        kernel_size (int, optional): Size of the kernel in the spatiotemporal convolution. Default is 25.
        pool_size (int, optional): Size of the pooling window. Default is 100.
        nhead (int, optional): Number of attention heads in the transformer. Default is 2.
    """

    def __init__(self, n_chans, n_outputs, n_times=1001, dropout=0.5, num_kernels=40, kernel_size=25, pool_size=100, nhead=2):
        super(CollapsedConformer, self).__init__()
        maxpool_out = (n_times - kernel_size + 1) // pool_size
        self.spatio_temporal = nn.Conv2d(
            n_chans, num_kernels, (1, kernel_size))
        self.pool = nn.AvgPool2d((1, pool_size))
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm2d(num_kernels)
        self.projection = nn.Conv2d(num_kernels, num_kernels, (1, 1))
        self.encoder_layers = nn.TransformerEncoderLayer(
            d_model=num_kernels, nhead=nhead, dim_feedforward=4*num_kernels, activation='gelu', batch_first=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(
            self.encoder_layers, num_layers=6, norm=nn.LayerNorm(num_kernels))
        hidden1_size = 256
        hidden2_size = 32
        self.fc = nn.Sequential(
            nn.Linear(num_kernels*maxpool_out, hidden1_size),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1_size, hidden2_size),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2_size, n_outputs)
        )

    def forward(self, x):
        x = torch.unsqueeze(x, dim=2)
        x = self.spatio_temporal(x)
        x = self.batch_norm(x)
        x = F.elu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.projection(x)
        x = x.squeeze(dim=2)
        x = rearrange(x, 'b d t -> b t d')
        x = self.transformer(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc(x)
        return x


class Conformer(nn.Module):
    """ 
    An implementation of the Conformer model from https://ieeexplore.ieee.org/document/9991178.

    This class represents a Conformer model, which is a deep learning model architecture for sequence classification tasks.
    It consists of several convolutional layers, a transformer encoder, and fully connected layers for classification.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        timepoints (int, optional): Number of timepoints in the input sequence. Defaults to 1001.
        dropout (float, optional): Dropout rate. Defaults to 0.5.
        num_kernels (int, optional): Number of kernels in the convolutional layers. Defaults to 40.
        kernel_size (int, optional): Size of the convolutional kernels. Defaults to 25.
        pool_size (int, optional): Size of the pooling window. Default is 100.
        nhead (int, optional): Number of attention heads in the transformer encoder. Defaults to 2.
    """

    def __init__(self, n_chans, n_outputs, n_times=1001, dropout=0.5, num_kernels=40, kernel_size=25, pool_size=100, nhead=2):
        super(Conformer, self).__init__()
        maxpool_out = (n_times - kernel_size + 1) // pool_size

        self.temporal = nn.Conv2d(1, num_kernels, (1, kernel_size))
        self.spatial = nn.Conv2d(num_kernels, num_kernels, (n_chans, 1))
        self.pool = nn.AvgPool2d((1, pool_size))
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm2d(num_kernels)
        self.projection = nn.Conv2d(num_kernels, num_kernels, (1, 1))
        self.encoder_layers = nn.TransformerEncoderLayer(
            d_model=num_kernels, nhead=nhead, dim_feedforward=4*num_kernels, activation='gelu', batch_first=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(
            self.encoder_layers, num_layers=6, norm=nn.LayerNorm(num_kernels))
        hidden1_size = 256
        hidden2_size = 32
        self.fc = nn.Sequential(
            nn.Linear(num_kernels*maxpool_out, hidden1_size),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1_size, hidden2_size),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2_size, n_outputs)
        )

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = self.temporal(x)
        x = self.spatial(x)
        x = F.elu(x)
        x = self.batch_norm(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.projection(x)
        x = x.squeeze(dim=2)
        x = rearrange(x, 'b d t -> b t d')
        x = self.transformer(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc(x)
        return x
        
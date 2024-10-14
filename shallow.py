from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F


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
        
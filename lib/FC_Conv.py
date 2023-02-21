import torch
import torch.nn as nn

import numpy as np

class FC_Conv(nn.Module):
    def __init__(self, szA, FC_sizes, conv_sizes, dev):
        super().__init__()
        
        im_sz = int(np.sqrt(szA[1]))
        
        # Set FC layers
        net_architecture = [nn.Linear(szA[0], FC_sizes[0], dtype=torch.double)]
        net_architecture.append(nn.Tanh())
        for i in range(len(FC_sizes)-1):
            net_architecture.append(nn.Linear(FC_sizes[i], FC_sizes[i+1], dtype=torch.double))
            net_architecture.append(nn.Tanh())
        net_architecture.append(nn.Linear(FC_sizes[-1], szA[1], dtype=torch.double))
        
        # Set Convolutional layers
        for j in range(len(conv_sizes)-1):
            if j == 0:
                net_architecture.append(nn.Tanh())
                net_architecture.append(nn.Unflatten(dim=1, unflattened_size=(1, im_sz, im_sz)))
                n_channels_i = 1
            else:
                n_channels_i = conv_sizes[j-1][1]
            pad_size_j = conv_sizes[j][0]
            n_channels_o = conv_sizes[j][1]
            net_architecture.append(nn.Conv2d(in_channels=n_channels_i, out_channels=n_channels_o, kernel_size=(2*pad_size_j+1), padding=pad_size_j, dtype=torch.double))
            net_architecture.append(nn.ReLU())

        # Set last Convolutional layer
        if len(conv_sizes) > 0:
            n_channels_i_fin = conv_sizes[-2][1]
            n_channels_o_fin = 1
            pad_size_fin = conv_sizes[-1]
            net_architecture.append(nn.ConvTranspose2d(in_channels=n_channels_i_fin, out_channels=n_channels_o_fin, kernel_size=(2*pad_size_fin+1), padding=pad_size_fin, dtype=torch.double))
            
        self.net = nn.Sequential(*net_architecture).to(dev)
        
    def forward(self, x):
        nElem = x.shape[1]
        return torch.reshape(torch.squeeze(self.net(x.T)), (nElem, -1)).T
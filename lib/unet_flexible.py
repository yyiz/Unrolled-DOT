import torch
import torch.nn as nn
import torch.nn.functional as F
import skimage
import numpy as np 
import scipy.io as sio
import torchvision.models as models
import math

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)
def dilconv3x3(in_channels, out_channels, stride=1,dilation=2):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, dilation=dilation, padding=2, bias=False)
# Residual block

def conv1x1(in_channels, out_channels, stride=2):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                     stride=stride, padding=0, bias=False)

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch,momentum=0.99),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch,momentum=0.99),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    
class double_conv2(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3,stride=2, padding=1),
            nn.BatchNorm2d(out_ch,momentum=0.99),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch,momentum=0.99),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x    

class resdouble_conv2(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(resdouble_conv2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3,stride=2, padding=1),
            nn.BatchNorm2d(out_ch,momentum=0.99),
            nn.ReLU(inplace=True)
        )
        self.resblock=ResidualBlock(out_ch,out_ch)

    def forward(self, x):
        x = self.resblock(self.conv(x))
        return x    
class dilresdouble_conv2(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(dilresdouble_conv2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3,stride=2, padding=1),
            nn.BatchNorm2d(out_ch,momentum=0.99),
            nn.ReLU(inplace=True)
        )
        self.resblock=DilatedResidualBlock(out_ch,out_ch)

    def forward(self, x):
        x = self.resblock(self.conv(x))
        return x    
class resdouble_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(resdouble_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch,momentum=0.99),
            nn.ReLU(inplace=True)
        )
        self.resblock=ResidualBlock(out_ch,out_ch)

    def forward(self, x):
        x = self.resblock(self.conv(x))
        return x    
class dilresdouble_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(dilresdouble_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch,momentum=0.99),
            nn.ReLU(inplace=True)
        )
        self.resblock=DilatedResidualBlock(out_ch,out_ch)

    def forward(self, x):
        x = self.resblock(self.conv(x))
        return x
class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            double_conv2(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class resdown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(resdown, self).__init__()
        self.mpconv = nn.Sequential(
            resdouble_conv2(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x
class dilresdown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(dilresdown, self).__init__()
        self.mpconv = nn.Sequential(
            dilresdouble_conv2(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x
class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class resup(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(resup, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = resdouble_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
class dilresup(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(dilresup, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = dilresdouble_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
class upnocat(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(upnocat, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x
class resupnocat(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(resupnocat, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv = resdouble_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x
class dilresupnocat(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(dilresupnocat, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv = dilresdouble_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3,padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x





class UNet(nn.Module):
    def __init__(self, n_filts=32, input_channels=1, bn_input=False):
        super(UNet, self).__init__()
        self.inc = inconv(input_channels, n_filts)
        self.down1 = down(n_filts, int(2*n_filts))
        self.down2 = down(int(2*n_filts), int(4*n_filts))
        self.down3 = down(int(4*n_filts), int(8*n_filts))
        self.down4 = down(int(8*n_filts), int(8*n_filts))
        self.up1 = up(int(16*n_filts), int(4*n_filts))
        self.up2 = up(int(8*n_filts), int(2*n_filts))
        self.up3 = up(int(4*n_filts), n_filts)
        self.up4 = up(int(2*n_filts), n_filts)
        self.outc = outconv(n_filts, input_channels)
        self.bn=nn.BatchNorm2d(input_channels,momentum=0.99)
        self.bn_input = bn_input
        

    def forward(self, x):
        if self.bn_input:
            x = self.bn((x))
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return torch.sigmoid(x)
    
    
def nearest2pow(n):
    p = len(bin(n)) - 3
    return 2**p
    
class UNet_wrapper(nn.Module):
    def __init__(self, imSz, nfilts=32, input_channels=1, bn_input=False):
        super().__init__()
        self.unet = UNet(nfilts, input_channels, bn_input) 
        self.imSz = imSz
        self.new_imlen = nearest2pow(self.imSz)
        if self.new_imlen != self.imSz:
            self.new_imlen *= 2
            self.pad_0 = math.floor((self.new_imlen-self.imSz)/2)
            self.pad_1 = math.ceil((self.new_imlen-self.imSz)/2)
            self.F_pad = nn.ZeroPad2d((self.pad_0, self.pad_1, self.pad_0, self.pad_1)) # zeropad function
        
    def get_params(self):
        return self.unet.parameters()
    
    def send2dev(self, dev):
        self.unet = self.unet.to(dev)
        
    def forward(self, x):
        # Reshape input
        x_reshape = torch.permute(torch.reshape(x, (self.imSz, self.imSz, -1)), (2,0,1))[:,None,:,:].float()
        
        # Zero-pad
        if self.new_imlen != self.imSz:
            x_reshape = self.F_pad(x_reshape)
        
        # Apply UNet
        unet_out = self.unet(x_reshape)
        
        # Crop
        if self.new_imlen != self.imSz:
            unet_out = unet_out[:,:,self.pad_0:(self.imSz+self.pad_0), self.pad_0:(self.imSz+self.pad_0)]
        
        return torch.reshape(unet_out, (-1, self.imSz*self.imSz)).T.double()
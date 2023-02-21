import torch
from collections import namedtuple
from torchvision import models

class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out



class VGG_Loss(torch.nn.Module):
    def __init__(self, imSz, loss_fn):
        super().__init__()
        self.vgg = Vgg16(requires_grad=False)
        self.imSz = imSz
        self.loss_fn = loss_fn
        
    def send2dev(self, dev):
        self.vgg = self.vgg.to(dev)
        
    def get_loss(self, ref, pred):
        ref_reshape = torch.permute(torch.reshape(ref, (self.imSz, self.imSz, -1)), (2,0,1))[:,None,:,:].repeat(1,3,1,1).float()
        pred_reshape = torch.permute(torch.reshape(pred, (self.imSz, self.imSz, -1)), (2,0,1))[:,None,:,:].repeat(1,3,1,1).float()
        vgg_truth = self.vgg(ref_reshape)
        vgg_pred = self.vgg(pred_reshape)
        return self.loss_fn(vgg_truth.relu2_2, vgg_pred.relu2_2) + self.loss_fn(vgg_truth.relu4_3, vgg_pred.relu4_3)
# -*- coding: utf-8 -*-
"""
InfoGAN model

__author__  = '{Jimmy Yeh}'
__email__   = '{marrch30@gmail.com}'
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GAN(nn.Module):
    def __init__(self, args):
        super(GAN, self).__init__()
        self.G = Generator(args)
        self.G.apply(weights_init)
        self.D = Discriminator(args)
        self.D.apply(weights_init)

    def save(self, filepath):
        state = {
            'gen_net': self.G.state_dict(),
            'dis_net': self.D.state_dict(),
        }
        torch.save(state, filepath)

    def load(self, filepath):
        state = torch.load(filepath)
        self.G.load_state_dict(state['gen_net'])
        self.D.load_state_dict(state['dis_net'])


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        torch.nn.init.constant_(m.bias, 0.0)

class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.main = nn.ModuleList()
        in_channel = args.z_dim + args.cc_dim + args.dc_dim
        for i, x in enumerate(args.layer_G):
            self.main.append(nn.ConvTranspose2d(in_channel, *x))
            in_channel = x[0]
            if i < len(args.layer_G)-1:
                if args.use_batchnorm:
                    self.main.append(nn.BatchNorm2d(in_channel))
                if args.use_relu:
                    self.main.append(nn.ReLU())
                else:
                    self.main.append(nn.Sigmoid())
            else:
                self.main.append(nn.Tanh())

    def forward(self,x):
        for layer in self.main:
            x = layer(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.args = args
        self.main = nn.ModuleList()
        in_channel = args.img_channel_num
        last = len(args.layer_D)-1
        for i, x in enumerate(args.layer_D):
            self.main.append(nn.Conv2d(in_channel, *x))
            in_channel = x[0]

            if i > 0 and i < last and args.use_batchnorm:
                self.main.append(nn.BatchNorm2d(in_channel))
            
            if i < last and args.use_relu:
                self.main.append(nn.LeakyReLU(0.2))
            else:
                self.main.append(nn.Sigmoid())
            
    def forward(self,x):
        for layer in self.main:
            x = layer(x)

        out = x.squeeze()
        # out[:,0] = F.sigmoid(out[:,0].clone())
        out = F.sigmoid(torch.cat( (out[:,0:0].clone(), out[:,1:]) ,-1))
        ft = self.args.cc_dim+1
        ed = self.args.cc_dim+1+self.args.dc_dim

        out[:,ft:ed] = F.softmax(out[:,ft:ed].clone(), dim=1)

        return x


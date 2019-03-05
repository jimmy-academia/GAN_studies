# -*- coding: utf-8 -*-
"""Function for DCGAN model class delaration
This module includes the DCGAN model

Example:
    model = GAN(args)
    # args should contain z_dim, layer_G, layer_D, use_batchnorm, use_relu
    model.G
    model.D

__author__  = '{Jimmy Yeh}'
__email__   = '{marrch30@gmail.com}'
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GAN(nn.Module):
    def __init__(self, args):
        super(GAN, self).__init__()
        if args.datatype=='mnist':
            self.G = simpleGen(args)
            self.D = simpleDis(args)
        # else:
        #     ### this part unchanged yet
        #     self.G = Generator(args)
        #     self.G.apply(weights_init)
        #     self.D = Discriminator(args)
        #     self.D.apply(weights_init)

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

class simpleGen(nn.Module):
    def __init__(self,args):
        super(simpleGen, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(args.z_dim + args.cc_dim + args.dc_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 128*7*7),
            nn.BatchNorm1d(128*7*7),
            nn.ReLU()
            )
        self.conv = nn.Sequential(
            # [-1, 128, 7, 7] -> [-1, 64, 14, 14]
            nn.ConvTranspose2d(128,64,4,2,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # -> [-1, 1, 28, 28]
            nn.ConvTranspose2d(64,1,4,2,1),
            nn.Tanh()
            )
    def forward(self,z):
        # [-1, z]

        z = self.fc(z)
        # [-1, 128*7*7] -> [-1, 128, 7, 7]
        z = z.view(-1, 128, 7, 7)
        out = self.conv(z)
        return out

class simpleDis(nn.Module):
    def __init__(self,args):
        super(simpleDis, self).__init__()
        self.cc_dim = args.cc_dim
        self.dc_dim = args.dc_dim

        self.conv = nn.Sequential(
            # [-1, 1, 28, 28] -> [-1, 64, 14, 14]
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),

            # [-1, 128, 7, 7]
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(128*7*7, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 1 + args.cc_dim + args.dc_dim)
        )
    def forward(self,x):
        # -> [-1, 128*7*7]
        x = self.conv(x)
        x = x.view(-1, 128*7*7)

        out = self.fc(x)
        # -> [-1, 1 + cc_dim + dc_dim]
        out[:, 0] = F.sigmoid(out[:, 0].clone())

        # Continuous Code Output = Value Itself
        # Discrete Code Output (Class -> Softmax)
        out[:, self.cc_dim + 1:self.cc_dim + 1 + self.dc_dim] = \
            F.softmax(out[:, self.cc_dim + 1:self.cc_dim + 1 + self.dc_dim].clone())

        return out







def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        torch.nn.init.constant_(m.bias, 0.0)

class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.main = nn.ModuleList()
        in_channel = args.z_dim
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
        self.main = nn.ModuleList()
        in_channel = args.img_channel_num
        for i, x in enumerate(args.layer_D):
            self.main.append(nn.Conv2d(in_channel, *x))
            in_channel = x[0]
            if i > 0 and i < len(args.layer_D)-1 and args.use_batchnorm:
                self.main.append(nn.BatchNorm2d(in_channel))
                
            if i < len(args.layer_D)-1 and args.use_relu:
                self.main.append(nn.LeakyReLU(0.2))
            else:
                self.main.append(nn.Sigmoid())
            
    def forward(self,x):
        for layer in self.main:
            x = layer(x)
        return x


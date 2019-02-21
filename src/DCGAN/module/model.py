######################################################
###  Created by Jimmy Yeh, Made In Taiwan          ###
###  for the sole purpose of being excellent       ###
###  What do people usually put here? this doesn't look pretty
######################################################

# description:
# in here generator model and discriminator model definitions
#
#       [usage]
#       G = generator()
#       D = discriminator()
#       

## currently written for MNIST dataset

import torch
import torch.nn as nn
import torch.nn.functional as F

class GAN(nn.Module):
    def __init__(self, img_channel_num):
        super(GAN, self).__init__()
        self.generator = Generator(img_channel_num)
        self.generator.apply(weights_init)
        self.discriminator = Discriminator(img_channel_num)
        self.generator.apply(weights_init)

    # def resume(self, checkpoint_path):
    #     self.generator.load_state_dict(torch.load(checkpoint_path+'gen.pk'))
    #     self.discriminator.load_state_dict(torch.load(checkpoint_path+'dis.pk'))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    def __init__(self, img_channel_num=1):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # (100)x1x1
            nn.ConvTranspose2d(100,64, 7, 1, 0, bias=False), ## layer size originally double
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. (64) x 7 x 7
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # state size. (32) x 14 x 14
            nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Tanh()
            # state size. (1) x 28 x 28 
        )

    def forward(self, input):
        output = self.main(input)
        return output

## pytorch nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding)
# output_width = (input_width -1) x stride - 2*padding + kernel_size - 1 + 1

#https://pytorch.org/docs/master/nn.html#torch.nn.ConvTranspose2d

class Discriminator(nn.Module):
    def __init__(self, img_channel_num=1):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (3) x 28 x 28    <------ size is different for different dataset!
            nn.Conv2d(img_channel_num, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (32) x 14 x 14
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64) x 7 x 7
            nn.Conv2d(64, 1, 7, 1, 0, bias=False),
            nn.BatchNorm2d(1),
            # state size. (1) x 1 x 1
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)


## pytorch nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding)
# output_width = (input_width - kernel_size + 2*padding)/stride + 1


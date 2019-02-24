# -*- coding: utf-8 -*-
"""Main training module for pytorch training on DCGAN model
This module puts together data and model, and perform various DCGAN 
trainings, including train, 
    
Example:
    trainer = Trainer(config, args, opt)
    trainer.train()

Todo:
    arithmetic operations, other functions needed for experiment
    show training history
    gif result
    grid result
    evaluation???
    

__author__  = '{Jimmy Yeh}'
__email__   = '{marrch30@gmail.com}'
"""

from module.model import GAN
from module.datafunc import make_dataset, make_dataloader

import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable


from tqdm import tqdm
from random import randint
from torchvision.utils import save_image

import os

class Trainer():
    def __init__(self, config, args, opt):
        self.model = GAN(args) 
        self.G_optimizer = optim.Adam(self.model.G.parameters(), lr=args.lr, betas=args.betas)
        self.D_optimizer = optim.Adam(self.model.D.parameters(), lr=args.lr, betas=args.betas)
        self.criterion = nn.BCELoss()
        self.config = config
        self.args = args
        self.opt = opt
        # self.check_directories()
        # self.badmodel = GAN(args)

    # def check_directories(self):
    #     for path in self.opt.dir_list:
    #         if not os.path.exists(path):
    #             os.makedirs(path)

    # def fastcheck(self, code='1'):
    #     print('fastcheck')
    #     dataset = make_dataset(self.config.data_dir_root, self.args)
    #     dataloader = make_dataloader(dataset, batch_size=self.args.batch_size)
    #     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #     if device =='cuda':
    #         self.model.cuda()

    #     print('train #', self.opt.epochs, 'D: %d, G:%d, save at %s'%
    #         (self.opt.k, self.opt.g, './dump/fastcheck'))
    #     self.train_one_epoch(dataloader, device, 1)


    def train(self):
        print('training for task:', self.config.taskname)
        dataset = make_dataset(self.config.data_dir_root, self.args)
        dataloader = make_dataloader(dataset, batch_size=self.args.batch_size)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device =='cuda':
            self.model.cuda()

        print('train #', self.opt.epochs, 'D: %d, G:%d, save at %s'%
            (self.opt.k, self.opt.g, self.opt.img_dir))
        for i in range(self.opt.epochs):
            self.train_one_epoch(dataloader, device, i)
            self.save_img_sample(device, self.opt.img_dir)

    def train_one_epoch(self, dataloader, device, epoch):
        real_label = 1
        fake_label = 0

        # pbar = tqdm(enumerate(dataloader))
        pbar = tqdm(dataloader)
        # for index, (inputs, __) in pbar:
        tmp = 0
        for inputs, __ in pbar:
            tmp = tmp+1
            if tmp==100:
                break
        # for inputs, __ in dataloader:
            batch_size = inputs.shape[0]
            label_real = Variable(torch.ones(batch_size).cuda())
            label_fake = Variable(torch.zeros(batch_size).cuda())

            # Discriminator 
            for __ in range(self.opt.k):
                inputs = inputs.to(device)
                inputs = Variable(inputs)
                out = self.model.D(inputs)
                out = out.view(-1)
                err_Dis_real = self.criterion(out, label_real)
                Dis_out = out.mean().item()

                z = torch.randn(batch_size, 100, 1, 1, device=device)
                z = Variable(z)

                fake_inputs = self.model.G(z)
                out = self.model.D(fake_inputs)
                out = out.view(-1)
                err_Dis_fake = self.criterion(out, label_fake)

                err_Dis = err_Dis_fake + err_Dis_real
                self.model.D.zero_grad()
                err_Dis.backward()
                self.D_optimizer.step()

            # Generator  maximize log(D(G(z)))
            for __ in range(self.opt.g):
                
                z = torch.randn(batch_size, 100, 1, 1, device=device)
                z = Variable(z)
                fake_inputs = self.model.G(z)

                out = self.model.D(fake_inputs)
                out = out.view(-1)
                Dis_gen_out = out.mean().item()

                err_Gen = self.criterion(out, label_real)
                
                self.model.D.zero_grad()
                self.model.G.zero_grad()
                err_Gen.backward()
                self.G_optimizer.step()

            # message = 'step[%d/%d] errD:%.4f,errG:%.4f,D(x):%.4f,D(G(z)):%.4f' \
            #     %(index, length, err_Dis.item(), err_Gen.item(), Dis_out, Dis_gen_out)
            message = 'errD:%.4f,errG:%.4f,D(x):%.4f,D(G(z)):%.4f' \
                %(err_Dis.item(), err_Gen.item(), Dis_out, Dis_gen_out)
            pbar.set_description(message)


    def save_img_sample(self, device, image_dir, code='gen_'):
        with torch.no_grad():
            z = torch.randn(1, 100, 1, 1, device=device)
            fake_img = self.model.G(z)
            save_image(fake_img.cpu(), image_dir+'/'+code+'.png')

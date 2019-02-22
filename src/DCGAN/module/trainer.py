######################################################
###  Created by Jimmy Yeh, Made In Taiwan          ###
###  for the sole purpose of being excellent       ###
###  What do people usually put here? this doesn't look pretty
######################################################

# description:
#     everything other than prepare dataset and create model
#     ex: training, 
#         show training history
#         show generated result
#         evaluating???
#         (and different configuration of above by config.py)

from module.model import GAN
from module.datafunc import make_dataset, make_dataloader

import torch
from torch import optim
import torch.nn as nn

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
        self.check_directories()

    def check_directories(self):
        for path in self.opt.dir_list:
            if not os.path.exists(path):
                os.makedirs(path)

    def train(self):
        # prepare data
        print('running train for', self.config.taskname)
        dataset = make_dataset(self.config.data_dir_root, self.args)
        dataloader = make_dataloader(dataset, batch_size=self.args.batch_size)
        # train 1:1

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device =='cuda':
            self.model.cuda()

        print('train #', self.opt.epochs, 'D: %d, G:%d, save at %s'%
            (self.opt.k, self.opt.g, self.opt.img_dir))
        self.opt.imgsample_epoch = self.opt.epochs//10
        for i in range(self.opt.epochs):   #200 --> 30min 2000--> 5hr
            self.train_one_epoch(dataloader, device, i)
            # if (i+1)%self.opt.imgsample_epoch == 0:
            self.save_img_sample(dataset, i, device, self.opt.img_dir)

    def train_one_epoch(self, dataloader, device, epoch):
        real_label = 1
        fake_label = 0
        
        # pbar = tqdm(dataloader)
        # for inputs, __ in pbar:
        for inputs, __ in dataloader:

            # Discriminator 1
            for __ in range(self.opt.k):
                inputs = inputs.to(device)
                batch_size = inputs.shape[0]
                label_real = torch.full((batch_size, ), real_label, device=device)
                out = self.model.D(inputs)
                out = out.view(-1)
                err_Dis_real = self.criterion(out, label_real)
                # err_Dis_real.backward(retain_graph=True)
                Dis_out = out.mean().item()

                z = torch.randn(batch_size, 100, 1, 1, device=device)
                fake_inputs = self.model.G(z)
                # label.fill_(fake_label)
                label_fake = torch.full((batch_size, ), fake_label, device=device)

            
                out = self.model.D(fake_inputs)
                out = out.view(-1)
                err_Dis_fake = self.criterion(out, label_fake)
                # err_Dis_fake.backward(retain_graph=True)

                err_Dis = err_Dis_fake + err_Dis_real
                self.model.D.zero_grad()
                err_Dis.backward()
                self.D_optimizer.step()

            # Generator  maximize log(D(G(z)))
            for __ in range(self.opt.g):
                
                z = torch.randn(batch_size, 100, 1, 1, device=device)
                fake_inputs = self.model.G(z)

                # label.fill_(real_label)
                out = self.model.D(fake_inputs)
                out = out.view(-1)
                Dis_gen_out = out.mean().item()

                err_Gen = self.criterion(out, label_real)
                # err_Gen.backward(retain_graph=True)
                
                self.model.D.zero_grad()
                self.model.G.zero_grad()
                err_Gen.backward()
                self.G_optimizer.step()

            # message = 'epoch [%d/%d] errD:%.4f,errG:%.4f,D(x):%.4f,D(G(z)):%.4f' \
            #     %(epoch, self.opt.epochs, err_Dis.item(), err_Gen.item(), Dis_out, Dis_gen_out)
            # # progress_bar(batch_idx, len(dataloader), message)
            # pbar.set_description(message)

    def save_img_sample(self, dataset, epoch, device, image_dir):
        # print('saving img')
        with torch.no_grad():
            # r = randint(0, len(dataset))
            # real_img = dataset[r][0][:]
            # save_image(real_img.cpu(), image_dir+'real_'+str(epoch)+'.png')
            
            z = torch.randn(1, 100, 1, 1, device=device)
            fake_img = self.model.G(z)
            save_image(fake_img.cpu(), image_dir+'/gen_'+str(epoch)+'.png')


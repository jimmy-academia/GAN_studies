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
from matplotlib import pyplot as plt 
plt.switch_backend('agg')

import numpy as np

import os


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

##### Helper Function for Math
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

# InfoGAN Function (Gaussian)
def gen_cc(n_size, dim):
    return torch.Tensor(np.random.randn(n_size, dim) * 0.5 + 0.0)

# InfoGAN Function (Multi-Nomial)
def gen_dc(n_size, dim):
    codes=[]
    code = np.zeros((n_size, dim))
    random_cate = np.random.randint(0, dim, n_size)
    code[range(n_size), random_cate] = 1
    codes.append(code)
    codes = np.concatenate(codes,1)
    return torch.Tensor(codes)



class Trainer():
    def __init__(self, config, args, opt):
        self.model = GAN(args) 
        self.G_optimizer = optim.Adam(self.model.G.parameters(), lr=args.lrG, betas=args.betas)
        self.D_optimizer = optim.Adam(self.model.D.parameters(), lr=args.lrD, betas=args.betas)
        self.criterion = nn.BCELoss()
        self.config = config
        self.args = args
        self.opt = opt

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.records = []

    def train(self):
        print('training for task:', self.config.taskname)
        dataset = make_dataset(self.config.data_dir_root, self.config.datatype, self.args.img_size)
        dataloader = make_dataloader(dataset, batch_size=self.args.batch_size)

        if self.device =='cuda':
            self.model.cuda()

        print('train #', self.opt.epochs, 'D: %d, G:%d, save at %s'%
            (self.opt.k, self.opt.g, self.opt.task_dir))

        # fixed_z = torch.randn(25, self.args.z_dim, 1, 1, device=self.device)
        for i in range(self.opt.epochs):
            epoch_records = self.train_one_epoch(dataloader)
            self.records.append(epoch_records)
            # self.save_img_sample(str(i))
            # self.save_fixed_grid_sample(fixed_z,'Epoch_'+str(i))
            self.save_loss_plot('Epoch_'+str(i))
        if self.opt.save_model:
            self.model.save(self.opt.model_filepath)

    def train_one_epoch(self, dataloader):
        pbar = tqdm(dataloader)
        epoch_records = []
        ## different for calebA (no label)
        for images, __ in pbar:
            batch_size = images.shape[0]
            label_real = Variable(torch.ones(batch_size).cuda())
            label_fake = Variable(torch.zeros(batch_size).cuda())
            
            images = to_variable(images)

            # Discriminator 
            for __ in range(1):                
                z = to_variable(torch.randn(batch_size, self.args.z_dim))
                cc = to_variable(gen_cc(batch_size, self.args.cc_dim))
                dc = to_variable(gen_cc(batch_size, self.args.dc_dim))
                noice = torch.cat((z, cc, dc), 1)
                noice = noice.view(noice.size(0), noice.size(1),1,1)
                fake_images = self.model.G(noice)
                d_out_real = self.model.D(images)#.view(-1)
                d_out_fake = self.model.D(fake_images)#.view(-1)
                # err_Dis_real = self.criterion(d_out_real, label_real)
                # err_Dis_fake = self.criterion(d_out_fake, label_fake)
                # d_loss_a = err_Dis_real + err_Dis_fake
                d_loss_a = -torch.mean(torch.log(d_out_real[:,0]) + torch.log(1 - d_out_fake[:,0]))
                output_cc = d_out_fake[:, 1:1+self.args.cc_dim]
                output_dc = d_out_fake[:, 1+self.args.cc_dim:]
                d_loss_cc = torch.mean((((output_cc - 0.0) / 0.5) ** 2))
                d_loss_dc = -(torch.mean(torch.sum(dc * output_dc, 1)) + torch.mean(torch.sum(dc * dc, 1)))

                d_loss = d_loss_a + self.args.continuous_weight * d_loss_cc + 1.0 * d_loss_dc

                self.model.D.zero_grad()
                d_loss.backward(retain_graph=True)
                self.D_optimizer.step()
                
                Dis_out = d_out_real.view(-1).mean().item()
                Dis_gen_out = d_out_fake.view(-1).mean().item()
            # Generator  maximize log(D(G(z)))
            for __ in range(1):
                g_loss_a = -torch.mean(torch.log(d_out_fake[:,0]))
                g_loss = g_loss_a + self.args.continuous_weight * d_loss_cc + 1.0 * d_loss_dc
                self.model.D.zero_grad() #???
                self.model.G.zero_grad()
                g_loss.backward()
                self.G_optimizer.step()

            batch_record = (d_loss.item(), g_loss.item(), Dis_out, Dis_gen_out)
            epoch_records.append(list(batch_record))
            message = 'errD:%.4f,errG:%.4f,D(x):%.4f,D(G(z)):%.4f'%batch_record
            pbar.set_description(message)

        epoch_records = np.mean(np.array(epoch_records),0)
        return epoch_records.tolist()

    # def save_img_sample(self, img_name='generated'):
    #     with torch.no_grad():
    #         z = torch.randn(1, 100, 1, 1, device=self.device)
    #         generated_img = self.model.G(z)
    #         save_image(generated_img.cpu(), self.opt.img_dir+'/'+img_name+'.png')


    # def save_fixed_grid_sample(self, fixed_z, img_name='generated'):
        
    #     with torch.no_grad():
    #         fixed_z = Variable(fixed_z)
    #         generated_ = self.model.G(fixed_z)
    #         def denorm(x):
    #             # is this needed???
    #             out = (x + 1) / 2
    #             return out.clamp(0, 1)
    #         genrated_ = denorm(generated_)

    #         n_rows = np.sqrt(fixed_z.size()[0]).astype(np.int32)
    #         n_cols = np.sqrt(fixed_z.size()[0]).astype(np.int32)
    #         fig, axes = plt.subplots(n_rows, n_cols, figsize=(5,5))
    #         for ax, img in zip(axes.flatten(), generated_):
    #             ax.axis('off')
    #             ax.imshow(img.cpu().data.view(self.args.img_size, self.args.img_size).numpy(), cmap='gray', aspect='equal')
    #         plt.subplots_adjust(wspace=0, hspace=0)
    #         # title = 'Epoch {0}'.format(num_epoch+1)
    #         fig.text(0.5, 0.04, img_name, ha='center')

    #         filepath = self.opt.task_dir+'/generated_imgs'
    #         if not os.path.exists(filepath):
    #             os.mkdir(filepath)
    #         plt.savefig(filepath+'/'+img_name+'.png')
    #         plt.close()

    def save_loss_plot(self, img_name='loss'):
        four_records = np.array(self.records).T
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.opt.epochs)
        ax.set_xlabel(img_name)

        ax.set_ylim(0, np.max(four_records[:2])*1.1)
        ax.set_ylabel('Loss values')
        ax.plot(four_records[0], label='D loss', color='#CC0000')
        ax.plot(four_records[1], label='G loss', color='#FF8000')
        ax.tick_params('y', colors = '#CC0000')
        ax.legend(loc='upper left')

        ax2 = ax.twinx()
        ax2.set_ylim(0, np.max(four_records[2:])*1.1)
        ax2.set_ylabel('D() values')
        ax2.plot(four_records[2], label='D(x)', color='#0080FF', linestyle=':')
        ax2.plot(four_records[3], label='D(G(z))', color='#808080', linestyle=':')
        ax2.tick_params('y', colors='k')
        ax2.legend(loc='upper right')

        
        filepath = self.opt.task_dir+'/loss_plots'
        if not os.path.exists(filepath):
            os.mkdir(filepath)
        plt.savefig(filepath+'/'+img_name+'.png')
        plt.close()





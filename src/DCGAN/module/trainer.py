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


class Trainer():
    def __init__(self):
        self.model = GAN()
        self.G_optimizer = optim.Adam(self.model.G.parameters(), lr=3e-4, betas=(0.5, 0.999))
        self.D_optimizer = optim.Adam(self.model.D.parameters(), lr=3e-4, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()

    def train(self):
        # prepare data
        dataset = make_dataset('/home/jimmy/datastore')
        dataloader = make_dataloader(dataset)
        # train 1:1

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device =='cuda':
            self.model.cuda()

        for i in range(20):
            self.train_one_epoch(dataloader, device)
            # if (i+1)%10 == 0:
            self.save_img_sample(dataset, i, device)

    def train_one_epoch(self, dataloader, device):
        real_label = 0
        fake_label = 1
        
        pbar = tqdm(dataloader)
        for inputs, __ in pbar:
            self.model.D.zero_grad()

            # Discriminator 1

            inputs = inputs.to(device)
            batch_size = inputs.shape[0]
            label = torch.full((batch_size, ), real_label, device=device)
            out = self.model.D(inputs)
            err_Dis_real = self.criterion(out, label)
            err_Dis_real.backward(retain_graph=True)
            Dis_out = out.mean().item()

            z = torch.randn(batch_size, 100, 1, 1, device=device)
            fake_inputs = self.model.G(z)
            label.fill_(fake_label)
        
            out = self.model.D(fake_inputs)
            err_Dis_fake = self.criterion(out, label)
            err_Dis_fake.backward(retain_graph=True)

            err_Dis = err_Dis_fake + err_Dis_real
            self.D_optimizer.step()

            # Generator  maximize log(D(G(z)))

            self.model.G.zero_grad()
            label.fill_(real_label)
            out = self.model.D(fake_inputs)
            Dis_gen_out = out.mean().item()

            err_Gen = self.criterion(out, label)
            err_Gen.backward(retain_graph=True)
            self.G_optimizer.step()

            message = 'errD:%.4f,errG:%.4f,D(x):%.4f,D(G(z)):%.4f' \
                %(err_Dis.item(), err_Gen.item(), Dis_out, Dis_gen_out)
            # progress_bar(batch_idx, len(dataloader), message)
            pbar.set_description(message)
    def save_img_sample(self, dataset, epoch, device, image_dir = 'img/'):
        with torch.no_grad():
            r = randint(0, len(dataset))
            real_img = dataset[r][0][:]
            save_image(real_img.cpu(), image_dir+'real_'+str(epoch)+'.png')
            
            z = torch.randn(1, 100, 1, 1, device=device)
            fake_img = self.model.G(z)
            save_image(fake_img.cpu(), image_dir+'gen_'+str(epoch)+'.png')


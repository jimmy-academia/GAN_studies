# -*- coding: utf-8 -*-
"""Function for data preparation 

This module includes functions used for data preparation. A typical usage would
be to call make_dataset() and make_dataloader() in trainer.py in when training. 

Example:
    dataset = make_dataset(self.config.data_dir_root, self.args)
    dataloader = make_dataloader(dataset, batch_size=self.args.batch_size)

Todo:
    * enlarge list of datasets to include cifar10, lsun, etc

__author__  = '{Jimmy Yeh}'
__email__   = '{marrch30@gmail.com}'
"""

import numpy as np
from torch.utils import data
from torchvision import datasets
from torchvision import transforms

#main function to be called
def make_dataset(data_dir_root, args):
    return MNIST(data_dir_root, args)

def make_dataloader(dataset, batch_size):
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

#list of datasets to use
def MNIST(data_dir_root, args):
    trans = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])

    dataset = datasets.MNIST(
        data_dir_root+'/mnist', train=True, download=True, transform=trans
    )
    return dataset

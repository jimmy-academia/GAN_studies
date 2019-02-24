######################################################
###  Created by Jimmy Yeh, Made In Taiwan          ###
###  for the sole purpose of being excellent       ###
###  What do people usually put here? this doesn't look pretty
######################################################

# description:
# in here is dataset(data_dir_root, datatype) for generating datasets
#     and data_loader(dataset) for creating dataloaders
#
#     [usage]
#     dataset =  dataset(
#         'path/to/parent_of_dataset_directory', 
#         'mnist' or 'cifar10')
#     dataloader = dataloader(dataset)



import numpy as np
from torch.utils import data
from torchvision import datasets
from torchvision import transforms

def make_dataset(data_dir_root, args):
    return MNIST(data_dir_root, args)

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

def CIFAR10(data_dir_root, train):
    trans = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # load dataset
    dataset = datasets.CIFAR10(
        data_dir_root+'/cifar10', train=train, download=True, transform=trans
    )
    return dataset

# def dataloader(dataset):
#     data_loader = torch.utils.data.DataLoader(
#         dataset, batch_size=batch_size, sampler=valid_sampler,
#         num_workers=num_workers, pin_memory=pin_memory,
#     )
#     return data_loader
# 
def make_dataloader(dataset, batch_size):
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
# -*- coding: utf-8 -*-
"""trains DCGAN on mnist for 20 epochs and creates grid display genarated from fixed noise
    
Example:
    in parent directory: python experiments/01basic_mnist.py

Todo:
    grid display
    move check directory to utils
    move default result directory to result
    

__author__  = '{Jimmy Yeh}'
__email__   = '{marrch30@gmail.com}'
"""

import sys
sys.path.append('..')
sys.path.append('.')

from module.trainer import Trainer
from module.config import configurations
from module.utils import check_directories

def main():
    config, args, opt = configurations('BASIC_CELEBA', 'celeba')
    check_directories(opt.dir_list)
    trainer = Trainer(config, args, opt)
    trainer.train()

if __name__ == '__main__':
    main()


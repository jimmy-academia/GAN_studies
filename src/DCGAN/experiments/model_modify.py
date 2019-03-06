# -*- coding: utf-8 -*-
"""trains DCGAN on mnist for 20 epochs and creates grid display genarated from fixed noise
    
Example:
    in parent directory: python experiments/02basic_mnist.py


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
    config, args, opt = configurations('MODLE_MODIFY_relu')
    check_directories(opt.dir_list)
    trainer = Trainer(config, args, opt)
    args.use_relu = False
    trainer.train()

    config, args, opt = configurations('MODLE_MODIFY_batchnorm')
    check_directories(opt.dir_list)
    trainer = Trainer(config, args, opt)
    args.use_batchnorm = False
    trainer.train()


if __name__ == '__main__':
    main()


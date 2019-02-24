import sys
sys.path.append('..')
sys.path.append('.')

from module.trainer import Trainer
from module.config import configurations
from module.utils import gpu_gauge, progress_bar

import torch.multiprocessing as mp 
import time


def subtask(config, args, opt):
    trainer = Trainer(config, args, opt)
    trainer.train()




def main():
    processes = []

    ## MEDIUM
    config, args, opt = configurations('mnist_medium')
    args.img_size = 32
    args.layer_G = [(512,4,1,0), (256,4,2,1), (128,4,2,1), (1,4,2,1)]
    args.layer_D = [(512,4,2,1), (256,4,2,1), (128,4,2,1), (1,4,1,0)]
    processes.append(mp.Process(target=subtask,\
        args=(config, args, opt)))

    ## SMALL
    config, args, opt = configurations('mnist_small')
    args.img_size = 28
    args.layer_G = [(256,7,1,0), (128,4,2,1), (1,4,2,1)]
    args.layer_D = [(256,4,2,1), (128,4,2,1), (1,7,1,0)]
    processes.append(mp.Process(target=subtask,\
        args=(config, args, opt)))
    
    ## LARGE
    config, args, opt = configurations('mnist_large')
    processes.append(mp.Process(target=subtask,\
        args=(config, args, opt)))
    # config, args, opt = configurations('mnist_1_3')
    # opt.k = 3
    # processes.append(mp.Process(target=subtask,\
    #     args=(config, args, opt)))
    
    # config, args, opt = configurations('mnist_1_7')
    # opt.k = 7
    # processes.append(mp.Process(target=subtask,\
    #     args=(config, args, opt)))


    # config, args, opt = configurations('mnist_3_1')
    # opt.k = 1
    # opt.g = 3
    # processes.append(mp.Process(target=subtask,\
    #     args=(config, args, opt)))

    # config, args, opt = configurations('mnist_7_1')
    # opt.g = 7
    # processes.append(mp.Process(target=subtask,\
    #     args=(config, args, opt)))

    memory = gpu_gauge()
    for process in processes:
        while not memory.available()>3000:
            time.sleep(5)
        print('hi')
        print(memory.available())
        process.start()
        process.join(0.1)
        time.sleep(15)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()


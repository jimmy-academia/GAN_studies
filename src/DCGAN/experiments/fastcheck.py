import sys
sys.path.append('..')

from module.trainer import Trainer
from module.config import configurations
from module.utils import gpu_gauge, progress_bar

def main():
    config, args, opt = configurations('fastcheck')

    trainer = Trainer(config, args, opt)
    trainer.fastcheck('test')
if __name__ == '__main__':
    main()


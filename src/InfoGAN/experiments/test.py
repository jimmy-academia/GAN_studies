import sys
sys.path.append('.')

from module.trainer import Trainer
from module.config import configurations

def main():
    config, args, opt = configurations('BASIC_MNIST')
    trainer = Trainer(config, args, opt)
    trainer.train()

if __name__ == '__main__':
    main()

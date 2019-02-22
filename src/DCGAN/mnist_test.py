import sys
sys.path.append('../module')

from module.trainer import Trainer
from module.config import configurations


def main():
	config, args, opt = configurations()
	trainer = Trainer(config, args, opt)
	trainer.train()

if __name__ == '__main__':
	main()


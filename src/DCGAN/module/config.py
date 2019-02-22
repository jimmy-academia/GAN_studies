import argparse
import time

## only those needed to be controlled from terminal will be passed by parser.
# other configurations will be controlled by simple class object

# I will use shorthand
# agrs for model parameters
# opt for training parameters
# config for other issues

# this will set everything to default status
def configurations():
	config, unparsed = parser.parse_known_args()
	if config.taskname is None:
		print('WARANING: USING COMPUTER TIME FOR TASKNAME')
		time.sleep(2.1)
		config.taskname = computer_time()

	print('taskname is:', config.taskname)
	
	args = model_param(config)
	opt = training_param(config)
	
	return config, args, opt

def computer_time():
	time_list = time.ctime().split()
	hr_min = ''.join(time_list[3].split(':')[:2])
	return '_'.join(time_list[1:3])+'_'+hr_min
	



# determine datatype (will effect model)
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--datatype', type=str, default='mnist',
					help='choose: mnist, cifar10, lsun, faces')
parser.add_argument('--taskname', type=str)


# permanent directories
dir_args = parser.add_argument_group('directories')
dir_args.add_argument('--data_dir_root', type=str, default='/home/jimmy/datastore',
	help='root for data download spot, \'/mnist/\' etc to be added')
dir_args.add_argument('--task_result_root', type=str, default='./dump')
# task related directories in training_param

class model_param():
	def __init__(self, config):
		self.img_channel_num = 1
		self.z_dim = 100
		self.layer_G = [(64,7,1,0), (32,4,2,1), (1,4,2,1)]
		self.layer_D = [(32,4,2,1), (64,4,2,1), (1,7,1,0)]
		self.use_batchnorm = True
		self.use_relu = True
		
		#trainer
		self.lr = 3e-4
		self.betas = (0.5, 0.999)
		self.refresh(config)

	def refresh(self, config):
		if config.datatype == 'cifar10':
			pass

# trainer setting
class training_param():
	def __init__(self, config):
		#train setting
		self.g = 1
		self.k = 1
		self.epochs = 4000
		self.imgsample_epoch = self.epochs//10

		# directories
		self.task_dir = config.task_result_root+'/'+config.taskname
		self.img_dir = self.task_dir+'/img'
		self.ckp_dir = self.task_dir+'/ckp'
		self.dir_list = [self.img_dir, self.ckp_dir]
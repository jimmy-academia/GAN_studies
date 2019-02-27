# -*- coding: utf-8 -*-
"""A parser and 2 class to settle all configurations

This module includes 1 parser and 2 class to settle all configurations.
Some values in class will depend on parser value (i.e. dir names following
taskname)

Notes:
	config => task related settings, directories
	args => model parameters
	opt => training options

Example:
	configurations('Name of this task')
Todo:
    * create function to refresh all connected values upon change.

__author__  = '{Jimmy Yeh}'
__email__   = '{marrch30@gmail.com}'
"""

import argparse
import time

def configurations(taskname=None):
	config, unparsed = parser.parse_known_args()

	def computer_time():
		time_list = time.ctime().split()
		hr_min = ''.join(time_list[3].split(':')[:2])
		return '_'.join(time_list[1:3])+'_'+hr_min
		
	if config.taskname is None:
		if taskname is None:
			print('WARANING: USING COMPUTER TIME FOR TASKNAME')
			time.sleep(2.1)
			config.taskname = 'T'+computer_time()
		else:
			config.taskname = taskname

	print('taskname is:', config.taskname)
	
	args = model_param(config)
	opt = training_param(config)
	return config, args, opt


# determine datatype (will effect model)
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--datatype', type=str, default='mnist',
					help='choose: mnist, cifar10, lsun, faces')
parser.add_argument('--taskname', type=str)


# permanent directories
dir_args = parser.add_argument_group('directories')
dir_args.add_argument('--data_dir_root', type=str, default='/home/jimmy/datastore',
	help='root for data download spot, \'/mnist/\' etc to be added')
dir_args.add_argument('--task_result_root', type=str, default='./output')
# task related directories in training_param

class model_param():
	def __init__(self, config):
		self.img_channel_num = 1
		self.z_dim = 100
		self.layer_G = [(1024,4,1,0), (512,4,2,1), (256,4,2,1), (128,4,2,1), (1,4,2,1)]
		self.layer_D = [(128,4,2,1), (256,4,2,1), (512,4,2,1), (1024,4,2,1), (1,4,1,0)]
		self.use_batchnorm = True
		self.use_relu = True
		
		#trainer
		self.lr = 0.0002
		self.betas = (0.5, 0.999)
		# self.refresh(config)

		#other
		self.img_size = 64
		self.batch_size = 128

	def refresh(self, config):
		if config.datatype == 'cifar10':
			pass

# trainer setting
class training_param():
	def __init__(self, config):
		#train setting
		self.g = 1
		self.k = 1
		self.epochs = 40

		# directories
		self.task_dir = config.task_result_root+'/'+config.taskname
		self.dir_list = [self.task_dir]

		self.save_model=False
		self.model_filepath = self.task_dir+'/model.t7'



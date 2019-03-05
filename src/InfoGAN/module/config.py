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

def computer_time():
	time_list = time.ctime().split()
	hr_min = ''.join(time_list[3].split(':')[:2])
	return '_'.join(time_list[1:3])+'_'+hr_min

class model_param():
	def __init__(self, config):
		if config.datatype =='mnist':
			self.datatype='mnist'
			self.img_channel_num = 1
		else:
			self.datatype='else'
			self.img_channel_num = 3
		self.img_size = 28
		self.z_dim = 62
		self.continuous_weight=0.5

		self.refresh()

		self.cc_dim = 1
		self.dc_dim = 10
		
		self.layer_G = [(1024,4,1,0), (512,4,2,1), (256,4,2,1), (128,4,2,1), (self.img_channel_num,4,2,1)]
		self.layer_D = [(128,4,2,1), (256,4,2,1), (512,4,2,1), (1024,4,2,1), (1,4,1,0)]
		self.use_batchnorm = True
		self.use_relu = True
		
		#trainer
		self.lrD = 0.0002
		self.lrG = 0.001
		self.betas = (0.5, 0.999)
		# self.refresh(config)

		#other
		self.batch_size = 16

	def refresh(self):

		if self.datatype=='else':
			self.img_size=64
			self.z_dim=128
			self.continuous_weight=1 # 0.1~0.5 for MNIST / 1.0 for CelebA

			

# trainer setting
class training_param():
	def __init__(self, config):
		#train setting
		self.g = 1
		self.k = 1
		self.epochs = 40
		self.refresh(config)

		# directories
		self.task_dir = config.task_result_root+'/'+config.taskname
		self.dir_list = [self.task_dir]

		self.save_model=False
		self.model_filepath = self.task_dir+'/model.t7'
	def refresh(self, config):
		if config.datatype!='mnist':
			self.epochs = 4


def configurations(taskname=None, datatype='mnist'):

	# determine datatype (will effect model)
	parser = argparse.ArgumentParser(description='INFOGAN')
	parser.add_argument('--taskname', type=str, default=taskname)
	parser.add_argument('--datatype', type=str, default=datatype,
						help='choose: mnist, cifar10, lsun, faces')


	# permanent directories
	dir_args = parser.add_argument_group('directories')
	dir_args.add_argument('--data_dir_root', type=str, default='~/datastore',
		help='root for data download spot, \'/mnist/\' etc to be added')
	#### new pytorch setting, mnist should not be added!!!
	dir_args.add_argument('--task_result_root', type=str, default='./output')
	# task related directories in training_param

	config, unparsed = parser.parse_known_args()

		
	if config.taskname is None and taskname is None:
		print('WARANING: USING COMPUTER TIME FOR TASKNAME')
		time.sleep(2.1)
		config.taskname = 'T'+computer_time()
	else:
		config.taskname = taskname

	print('taskname is:', config.taskname)
	print('datatype is:', config.datatype)
	args = model_param(config)
	opt = training_param(config)
	return config, args, opt



import myTorch
from myTorch.utils import *
import torch.optim as optim


def copy_task_RNN():

	config = MyContainer()

	# model specific details
	config.model = "TARDIS"
	config.input_size = 9
	config.output_size = 9
	config.num_layers = 2
	config.layer_size = [256, 256]

	# optimization specific details
	config.optim_algo = optim.RMSprop
	config.momentum = 0.9
	config.grad_clip = [-10, 10]
	config.l_rate = 0.00003
	config.max_steps = 1000000
	config.rseed = 5
	config.use_gpu = False


	# task specific details
	config.num_bits = 8
	config.min_len = 1
	config.max_len = 20

	# saving details
	config.use_tflogger = True
	config.tflogdir = "/mnt/data/chinna/output/copyRNN/tflog/p2/"
	config.out_folder = "/mnt/data/chinna/output/copyRNN/p2/"

	return config

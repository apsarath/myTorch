import myTorch
from myTorch.utils import *

def copy_task_RNN():

	config = MyContainer()

	# model specific details
	config.model = "LSTM"
	config.num_layers = 3
	config.layer_size = [256, 256, 256]

	# optimization specific details
	config.optim = "RMSProp"
	config.momentum = 0.9
	config.grad_clip = [-10, 10]
	config.l_rate = 0.0001
	config.max_steps = 1000000

	# task specific details
	config.num_bits = 8
	config.min_len = 1
	config.max_len = 20

	return config
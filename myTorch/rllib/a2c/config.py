import myTorch
from myTorch.utils import *

def cartpole():

	config = MyContainer()


	config.exp_name = "a2c"

	config.env_name = "CartPole-v0"
	config.policy_type = "FeedForward" # Valid options : LSTM, GRU, FeedForward

	config.discount_rate = 0.90
	config.ent_coef = 0.001
	config.vf_coef = 0.5

	config.global_num_steps = 100000
	config.num_env = 10
	config.num_steps_per_upd = 10
	config.test_freq = 1
	config.test_per_iter = 1

	config.save_freq = 500
	config.sliding_wsize = 30
	config.seed = 1234
	config.use_gpu = True
	config.train_dir = "outputs/"
	config.logger_dir = "logs/"
	config.use_tflogger = True
	config.backup_logger = False
	config.force_restart = False

	# optimizer params
	config.optim_name = "RMSprop" # valid optimizer names : Adadelta, Adagrad, Adam, RMSprop, SGD
	config.lr = 1e-3
	config.rho = 0.9
	config.eps = 1e-8
	config.weight_decay = 0
	config.lr_decay = 0
	config.beta_0 = 0.9
	config.beta_1 = 0.999
	config.alpha = 0.99
	config.momentum = 0
	config.centered = False
	config.dampening = 0
	config.nesterov = False 
	config.grad_clip_min = -1
	config.grad_clip_max = 1

	return config


def cartpole_image():
	config = cartpole()
	config.env_name = "CartPole-v0-image"
	return config

def blocksworld_matrix():
	config = cartpole()
	config.env_name = "blocksworld_matrix"
	config.policy_type = "GRU" # Valid options : LSTM, GRU, FeedForward
	config.global_num_steps = int(100e6)
	config.discount_rate = 0.99
	config.lr = 0.00025
	config.eps = 1e-6
	config.test_freq = 1000
	config.test_per_iter = 10
	return config

def gym_minigrid_image():
	config = cartpole()
	config.env_name = "MiniGrid-DoorKey-5x5-v0"
	return config


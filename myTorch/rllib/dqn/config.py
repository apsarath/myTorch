import myTorch
from myTorch.utils import *

def cartpole():

	config = MyContainer()

	config.exp_name = "dqn"

	config.env_name = "CartPole-v0"

	config.epsilon_end = 0.1
	config.epsilon_start = 0.1
	config.epsilon_end_t = 1
	config.learn_start = 1
	config.discount_rate = 0.99
	config.target_net_soft_update = True
	config.target_net_update_freq = 1
	config.target_net_update_fraction = 0.05

	config.num_iterations = 10000
	config.episodes_per_iter = 1
	config.test_freq = 1
	config.test_per_iter = 1
	config.updates_per_iter = 5
	config.batch_size = 64
	config.replay_buffer_size = 1e5
	config.replay_compress = False

	config.save_freq = 500
	config.sliding_wsize = 30
	config.seed = 1234
	config.use_gpu = True
	config.train_dir = "outputs/"
	config.logger_dir = "logs/"
	config.use_tflogger = True
	config.backup_logger = False
	config.force_restart = True

	# optimizer params
	config.optim_name = "Adam" # valid optimizer names : Adadelta, Adagrad, Adam, RMSprop, SGD
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
	config.grad_clip_min = None
	config.grad_clip_max = None
	return config

def cartpole_image():

	config = MyContainer()

	config.exp_name = "dqn"

	config.env_name = "CartPole-v0-image"

	config.epsilon_end = 0.05
	config.epsilon_start = 0.9
	config.epsilon_end_t = 2000
	config.learn_start = 1
	config.discount_rate = 0.999
	config.target_net_soft_update = True
	config.target_net_update_freq = 1
	config.target_net_update_fraction = 1

	config.num_iterations = 10000
	config.episodes_per_iter = 1
	config.test_freq = 1
	config.test_per_iter = 1
	config.updates_per_iter = 1
	config.batch_size = 128
	config.replay_buffer_size = 1e4
	config.replay_compress = False

	config.save_freq = 500
	config.sliding_wsize = 30
	config.seed = 1234
	config.use_gpu = False
	config.train_dir = "outputs/"
	config.logger_dir = "logs/"
	config.use_tflogger = True
	config.backup_logger = False
	config.force_restart = True

	# optimizer params
	config.optim_name = "RMSprop" # valid optimizer names : Adadelta, Adagrad, Adam, RMSprop, SGD
	config.lr = 1e-2
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


def blocksworld():
	config = cartpole_image()
	config.env_name = "blocksworld_matrix"
	config.lr = 0.000025
	config.eps = 1e-6
	config.use_gpu = True
	config.num_iterations = int(100e6)
	return config


def mazebase_array():

	config = MyContainer()

	config.exp_name = "dqn"

	config.env_name = "SingleMazeInstr-v0"

	config.epsilon_end = 0.05
	config.epsilon_start = 1
	config.epsilon_end_t = 100000
	config.learn_start = 10000
	config.discount_rate = 0.999
	config.target_net_soft_update = False
	config.target_net_update_freq = 10000
	config.target_net_update_fraction = 1

	config.num_iterations = 1000000
	config.episodes_per_iter = 1
	config.test_freq = 1000
	config.test_per_iter = 100
	config.updates_per_iter = 5
	config.batch_size = 64
	config.replay_buffer_size = 1e5
	config.replay_compress = False

	config.save_freq = 500
	config.sliding_wsize = 30
	config.seed = 1234
	config.use_gpu = False
	config.train_dir = "outputs/"
	config.logger_dir = "logs/"
	config.use_tflogger = True
	config.backup_logger = False
	config.force_restart = True

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
import myTorch
from myTorch.utils import *

def cartpole():

	config = MyContainer()


	config.exp_name = "a2c"

	config.env_name = "CartPole-v0"
	config.policy_type = "FeedForward" # Valid options : LSTM, GRU, FeedForward

	config.discount_rate = 0.99
	config.ent_coef = 1.0
	config.vf_coef = 1.0

	config.global_num_steps = 10000
	config.num_env = 1
	config.num_steps_per_upd = 10

	config.num_iterations = 10000
	config.test_freq = 1
	config.test_per_iter = 1

	config.save_freq = 500
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
	config = cartpole()
	config.env_name = "CartPole-v0-image"
	return config

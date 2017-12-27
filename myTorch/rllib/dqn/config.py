import myTorch
from myTorch.utils import *

def dqn():

	config = MyContainer()


	config.exp_name = "dqn"

	config.env_name = "CartPole-v0"
	config.qnet = "FeedForward"

	config.epsilon_end = 0.1
	config.epsilon_start = 1.0
	config.epsilon__end_t = 1e5
	config.learn_start = 50000
	config.discount_rate = 0.99
	config.target_net_update_freq = 10000

	config.num_iterations = 100000
	config.episodes_per_iter = 1
	config.test_freq = 10000
	config.test_per_iter = 100
	config.updates_per_iter = 5
	config.batch_size = 64
	config.replay_buffer_size = 1e5
	config.replay_compress = False

	config.save_freq = 1000
	config.sliding_wsize = 30
	config.seed = 786
	config.use_gpu = False
	config.train_dir = "outputs/"
	config.logger_dir = "logs/"
	config.use_tflogger = True

	# optimizer params
	config.optim_name = "RMSprop" # valid optimizer names : Adadelta, Adagrad, Adam, RMSprop, SGD
	config.lr = 1e-3
	config.rho = 0.9
	config.eps = 1e-6
	config.weight_decay = 0
	config.lr_decay = 0
	config.beta_0 = 0.9
	config.beta_1 = 0.999
	config.alpha = 0.99
	config.momentum = 0
	config.centered = False
	config.dampening = 0
	config.nesterov = False 

	return config
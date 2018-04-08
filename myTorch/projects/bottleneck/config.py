import myTorch
from myTorch.utils import *

def home_world():

	config = MyContainer()


	config.exp_name = "dqn"

	config.env_name = "home_world"

	config.epsilon_end = 0.2
	config.epsilon_start = 1
	config.epsilon_end_t = int(1e5)
	config.learn_start = 1
	config.discount_rate = 0.50
	config.target_net_soft_update = True
	config.target_net_update_freq = 1
	config.target_net_update_fraction = 1.0

	config.num_iterations = 12#int(1e5)
	config.episodes_per_iter = 5
	config.test_freq = 1
	config.test_per_iter = 50
	config.updates_per_iter = int(1)#50.0/4)
	config.batch_size = 64
	config.replay_buffer_size = int(5e3)
	config.replay_compress = False

	config.save_freq = 10
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
	config.lr = 0.0005
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

	# w2v location
	config.embedding_loc = "/mnt/data/chinna/data/glove/glove.6B.50d.txt"
	config.patience = 5
	config.num_new_samples = 2000
	config.cluster_num = 32
	return config

def fantasy_world():
	config = home_world()
	config.env_name = "fantasy_world"
	config.episodes_per_iter = 1
	config.test_freq = 10
	config.test_per_iter = 5
	config.updates_per_iter = int(50/4)
	return config

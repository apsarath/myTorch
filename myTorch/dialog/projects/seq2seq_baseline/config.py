import myTorch
from myTorch.utils import *

def commong_config():

	config = MyContainer()
    # training params
	config.exp_name = "seq2seq_baseline"
	config.num_iterations = 10000
	config.batch_size = 64
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

def opus():
    config = commong_config()

    # dataset params
    config.dataset_name = "opus"

    config.toy_mode = False
    config.train_valid_split = 0.8
    config.eou = "<eou>"
    config.go = "<go>"
    config.unk = "<unk>"
    config.num = "<num>"
    config.sentence_len_cut_off = 25
    config.min_sent_len = 6
    return config



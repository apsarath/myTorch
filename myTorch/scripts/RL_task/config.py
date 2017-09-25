import myTorch
from myTorch.utils import *
import torch.optim as optim

def cartpole_reinforce():

	config = MyContainer()

	config.env_name = "CartPole-v0"

	# model specific details
	config.hidden_size = 128

	# RL algo details
	config.alg_param = {}
	config.alg_param["use_etpy"] = True
	config.alg_param["etpy_coeff"] = 0.0001
	config.num_steps = 1000
	config.num_episodes = 10000
	config.gamma = 0.99

	# optimization specific details
	config.optim_algo = optim.RMSprop
	config.momentum = 0.9
	config.l_rate = 0.00003
	config.max_steps = 1000000
	config.rseed = 5
	config.use_gpu = False


	# saving details
	config.use_tflogger = True
	config.display = False
	config.tflogdir = "/mnt/data/sarath/output/cartpole/tflog/p1/"
	config.out_folder = "/mnt/data/sarath/output/cartpole/p1/"
	config.video_folder = "/mnt/data/sarath/output/cartpole/p1/video/"

	return config
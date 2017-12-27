import myTorch
from myTorch.utils import *
import torch.optim as optim

def dqn():

	config = MyContainer()

	config.epsilon_end = 0.1
	config.epsilon_start = 1.0
	config.epsilon__end_t = 1e5
	config.learn_start = 50000
	config.discount_rate = 0.99
	config.target_net_update_freq = 10000
	config.learning_rate = 1e-3

	config.exp_name = "dqn"


	return config
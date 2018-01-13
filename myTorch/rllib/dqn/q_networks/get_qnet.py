import myTorch
from myTorch.environment import GymEnvironment
from myTorch.rllib.dqn.q_networks import *

def get_qnet(env_name, obs_dim, action_dim, use_gpu):

	if env_name == "CartPole-v0" or env_name == "CartPole-v1":
		return FeedForwardCartPole(obs_dim, action_dim, use_gpu)
	elif env_name == "CartPole-v0-image" or env_name == "CartPole-v1-image":
		return ConvCartPole(obs_dim, action_dim, use_gpu)
	elif env_name == "blocksworld_matrix":
		return ConvBlocksWorld(obs_dim, action_dim, use_gpu)
	else:
		assert("unsupported environment : {}".format(env_name))

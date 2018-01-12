import myTorch
from myTorch.environment import GymEnvironment
from myTorch.rllib.a2c.a2c_networks import *

def get_a2cnet(env_name, obs_dim, action_dim, use_gpu, policy_type):

	if env_name == "CartPole-v0" or env_name == "CartPole-v1":
		if policy_type == "FeedForward":
			return FeedForwardCartPole(obs_dim, action_dim, use_gpu)
		elif policy_type == "LSTM" or policy_type == "GRU":
			return RecurrentCartPole(obs_dim, action_dim, use_gpu, policy_type)

	elif env_name == "CartPole-v0-image" or env_name == "CartPole-v1-image":
		if policy_type == "FeedForward":
			return ConvCartPole(obs_dim, action_dim, use_gpu)
		elif policy_type == "LSTM" or policy_type == "GRU":
			return RecurrentCartPole(obs_dim, action_dim, use_gpu, policy_type)

	elif env_name == "blocksworld_matrix":
		return FeedForwardBlocksWorldMatrix(obs_dim, action_dim, use_gpu)

	else:
		assert("unsupported environment : {}".format(env_name))

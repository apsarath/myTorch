import myTorch
from myTorch.environment import GymEnvironment
from myTorch.rllib.dqn.q_networks import *

def get_qnet(env_name, obs_dim, action_dim, use_gpu):

	if env_name == "CartPole-v0":
		return FeedForwardCartPoleV0(obs_dim, action_dim, use_gpu)
	elif env_name == "CartPole-v1":
		return ConvCartPoleV1(obs_dim, action_dim, use_gpu)
	else:
		assert("unsupported environment : {}".format(env_name))

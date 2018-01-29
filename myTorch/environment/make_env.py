import myTorch
from environment.Blocksworld import *
from environment import GymEnvironment, CartPoleImage, BlocksEnvironment, BlocksWorldMatrixEnv, GymMiniGrid

def make_environment(env_name):

	if env_name == "CartPole-v0" or env_name == "CartPole-v1":
		return GymEnvironment(env_name)
	elif env_name == "CartPole-v0-image" or env_name == "CartPole-v1-image":
		return CartPoleImage(env_name.replace("-image",""))
	elif env_name == "blocksworld_none":
		return BlocksEnvironment()
	elif env_name == "blocksworld_matrix":
		return BlocksWorldMatrixEnv()
	elif "MiniGrid" in env_name:
		import gym_minigrid
		return GymMiniGrid(env_name)
	else:
		assert("unsupported environment : {}".format(env_name))

if __name__=="__main__":
	env = make_environment("CartPole-v0")
	obs, legal_moves = env.reset()
	import pdb; pdb.set_trace()

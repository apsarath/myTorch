import myTorch
from myTorch.environment.Blocksworld import *
from myTorch.environment import GymEnvironment, CartPoleImage, BlocksEnvironment, BlocksWorldMatrixEnv, HomeWorldEnv


def make_environment(env_name, game_dir=None, mode="train", is_one_hot_world=False):

	if env_name == "CartPole-v0" or env_name == "CartPole-v1":
		return GymEnvironment(env_name)
	elif env_name == "CartPole-v0-image" or env_name == "CartPole-v1-image":
		return CartPoleImage(env_name.replace("-image",""))
	elif env_name == "MazeBaseInstr-v0":
		return MazeBaseEnvironment("MazeBaseInstr-v0")
	elif env_name == "SingleMazeInstr-v0":
		return MazeBaseEnvironment("SingleMazeInstr-v0")
	elif env_name == "blocksworld_none":
		return BlocksEnvironment()
	elif env_name == "blocksworld_matrix":
		env = BlocksWorldMatrixEnv(game_dir, mode=mode, is_one_hot_world=is_one_hot_world)
		return env
	elif env_name == "memory":
		env = MemoryEnvironment()
	elif env_name == "home_world":
		env = HomeWorldEnv()
		return env
	else:
		assert("unsupported environment : {}".format(env_name))

if __name__=="__main__":
	env = make_environment("blocksworld_matrix")
	obs, legal_moves = env.reset()
	import pdb; pdb.set_trace()

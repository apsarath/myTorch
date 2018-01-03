import myTorch
from myTorch.environment import GymEnvironment, CartPoleImage

def make_environment(env_name):

	if env_name == "CartPole-v0" or env_name == "CartPole-v1":

		env = GymEnvironment(env_name)
		return env
	elif env_name == "CartPole-v0-image" or env_name == "CartPole-v1-image":
		env = CartPoleImage(env_name)
	else:
		assert("unsupported environment : {}".format(env_name))


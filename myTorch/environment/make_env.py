import myTorch
from myTorch.environment import GymEnvironment

def make_environment(env_name):

	if env_name == "CartPole-v0":

		env = GymEnvironment(env_name)
		return env

	else:

		assert("unsupported environment : {}".format(env_name))


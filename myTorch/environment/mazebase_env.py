import gym
import myTorch
from myTorch.environment import EnivironmentBase
import numpy as np

class MazeBaseEnvironment(EnivironmentBase):

	def __init__(self, env_name):
		import mazebaseinstr
		self._env = gym.make('MazeBaseInstr-v0')
		self._action_dim = self._env.action_space.n
		self._legal_moves = np.arange(self._action_dim)
		self._obs_dim = self._env.observation_space.shape

	@property
	def action_dim(self):
		return self._action_dim

	@property
	def obs_dim(self):
		return self._obs_dim

	def reset(self):
		obs = self._env.reset()
		legal_moves = np.arange(self._action_dim)
		return obs, self._legal_moves

	def step(self, action):
		obs, reward, done, _ = self._env.step(action)
		return obs, self._legal_moves, reward, done

	def render(self, mode='rgb_array'):
		pass

	def seed(self, seed):
		self._env._seed(seed=seed)

	def get_random_state(self):
		pass

	def set_random_state(self, state):
		pass

	def save(self, save_dir):
		return
		
	def load(self, save_dir):
		return  


if __name__=="__main__":
	env = MazeBaseEnvironment("CartPole-v0")
	import pdb; pdb.set_trace()

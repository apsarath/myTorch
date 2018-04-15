import gym
import myTorch
import _pickle as pickle
from myTorch.environment import EnivironmentBase
import numpy as np
from myTorch.environment.text_world import HomeWorldGame

class HomeWorldEnv(EnivironmentBase):

	def __init__(self, numpy_rng=np.random.RandomState(1234), env_name="home_world", max_sent_len=30, max_epi_len=10, verbose=False, user_id=1):
		self._env_name = env_name
		self._max_sent_len = max_sent_len
		self._numpy_rng = numpy_rng
		self._max_epi_len = max_epi_len
		self._verbose = verbose
		if self._env_name == "home_world":
			self._game = HomeWorldGame(self._numpy_rng, self._max_sent_len, user_id=user_id)
		self._action_dim = self._game.num_actions
		self._legal_moves = np.arange(self._action_dim)

	@property
	def action_dim(self):
		return self._action_dim

	@property
	def obs_dim(self):
		return (2, self._max_sent_len)

	@property
	def max_episode_len(self):
		return self._max_epi_len

	@property
	def vocab(self):
		return (self._game.vocab, self._game.ivocab)

	def reset(self):
		obs, text_obs = self._game.reset()
		self._num_steps = 0
		if self._verbose: return text_obs
		return np.stack(obs, axis=0), self._legal_moves

	def step(self, action):
		self._num_steps += 1
		obs, reward, done, text_obs = self._game.step(action)
		if self._num_steps >= self._max_epi_len:
			done = True
		if self._verbose: return text_obs, reward, done
		return np.stack(obs, axis=0), self._legal_moves, reward, done

	def render(self, mode='rgb_array'):
		pass

	def seed(self, seed):
		self._numpy_rng = np.random.RandomState(seed)
		self._game.seed(seed)

	def get_random_state(self):
		pass

	def set_random_state(self, state):
		pass

	def save(self, filename):
		with open(filename, "wb") as f:
			pickle.dump(self._game.vocab, f)

	def load(self, filename):
		with open(filename, "rb") as f:
			self._game.load_vocab(pickle.load(f))


if __name__=="__main__":
	numpy_rng = np.random.RandomState(1234)
	env = HomeWorldEnv(numpy_rng, verbose=True)
	import pdb; pdb.set_trace()

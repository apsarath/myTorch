from myTorch.environment import EnvironmentBase
import numpy as np
import os

os.sys.path.append('\Users\prasanna\Documents\Blocksworld')
from blocksworldEnv import *

class BlocksEnvironment_colors(EnvironmentBase):
    def __init__(self):
        self._env = blocksworldEnv.Environment('colors')
        self._action_dim = len(self._env.actions)
		self._legal_moves = np.arange(self._action_dim)
		self._obs_dim = len(self._env.observation_space)

    	@property
    	def action_dim(self):
    		return self._action_dim

    	@property
    	def obs_dim(self):
    		return self._obs_dim

    	def reset(self):
    		obs = self._env.reset('colors','./Blocksworld/problem.txt',0)
    		legal_moves = np.arange(self._action_dim)
    		return obs, legal_moves

    	def step(self, action):
    		done, reward, obs = self._env.step(action)
    		return [curr_state, target], self._legal_moves, reward, done

    	def render(self, mode='rgb_array'):
    		pass

    	def seed(self, seed):
            pass

    	def get_random_state(self):
    		pass

    	def set_random_state(self, state):
    		pass

    	def save(self, save_dir):
    		return

    	def load(self, save_dir):
    		return

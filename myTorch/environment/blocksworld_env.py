import myTorch
from myTorch.environment import EnivironmentBase
import numpy as np
import os
from myTorch.environment.Blocksworld import *

class BlocksEnvironment(EnivironmentBase):
    def __init__(self):
        self._env = blocksworldEnv.Environment('None', "../../environment/Blocksworld/images/")
        self._action_dim = len(self._env.actions)
        self._legal_moves = np.arange(self._action_dim)
        self._obs_dim = self._env.observation_space

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def obs_dim(self):
        return self._obs_dim

    def reset(self):
        obs = self._env.reset('None','../../environment/Blocksworld/problem.txt',0)
        return obs, self._legal_moves

    def step(self, action):
        done, reward, obs = self._env.step(action)
        return obs, self._legal_moves, reward, done

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

if __name__=="__main__":
	env = BlocksEnvironment()
	import pdb; pdb.set_trace()

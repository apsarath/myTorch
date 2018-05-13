from myTorch.environment import EnivironmentBase
import numpy as np
from myTorch.utils import MyContainer


class MemoryEnvironment(EnivironmentBase):

    def __init__(self, plot_len=10, time_lag=5, seed=5):

        self._state = MyContainer()

        self._state.plot_len = plot_len
        self._state.time_lag = time_lag
        self._state.action_dim = 3
        self._state.legal_moves = np.arange(self._state.action_dim)
        self._state.obs_dim = plot_len + 1
        self._state.rng = np.random.RandomState(seed)

    @property
    def action_dim(self):
        return self._state.action_dim

    @property
    def obs_dim(self):
        return self._state.obs_dim

    def reset(self):
        self._final_reward = -5
        self._plot = np.zeros(self._state.plot_len + 1)
        self._color = self._state.rng.randint(4,6)
        self._rlocs = self._state.rng.choice(self._state.plot_len, 3, replace=False)
        
        self._plot[self._rlocs[0]] = 2
        self._plot[self._rlocs[1]] = 3
        self._agent_loc = self._rlocs[2]
        self._plot[self._agent_loc] = 1
        self._plot[-1] = self._color
        self._num_steps = 0
        self._touch2 = False
        self._touch3 = False
        return self._plot/10, self._state.legal_moves

    def step(self, action):
        self._num_steps += 1
        self._plot[-1] = 0

        if action == 0 and self._agent_loc > 0:
            self._agent_loc_new = self._agent_loc - 1
            self._plot[self._agent_loc] = 0
            self._plot[self._agent_loc_new] = 1
            self._agent_loc = self._agent_loc_new

        elif action == 1 and self._agent_loc < self._state.plot_len - 1:
            self._agent_loc_new = self._agent_loc + 1
            self._plot[self._agent_loc] = 0
            self._plot[self._agent_loc_new] = 1
            self._agent_loc = self._agent_loc_new

        done = False
        reward = 0#-0.1
        if self._num_steps == 12:
            done = True
            if not self._touch2 and not self._touch3:
                reward = -10

        if self._agent_loc == self._rlocs[0] and action == 2 and self._touch2 == False:
            self._touch2 = True
            self._touch3 = True
            reward = 5 if self._color == 4 else -5
        elif self._agent_loc == self._rlocs[1] and action == 2 and self._touch3 == False:
            self._touch3 = True
            self._touch2 = True
            reward = -5 if self._color == 4 else 5

        return self._plot/10, self._state.legal_moves, reward, done

    def render(self, mode='rgb_array'):
        pass

    def seed(self, seed):
        self._state.rng = np.random.RandomState(seed)

    def get_random_state(self):
        pass

    def set_random_state(self, state):
        pass

    def save(self, save_dir):
        self._state.save(save_dir)

    def load(self, save_dir):
        self._state.load(save_dir)


if __name__ == "__main__":
    env = MemoryEnvironment()
    import pdb;

    pdb.set_trace()

from myTorch.environment import EnivironmentBase
import numpy as np
from myTorch.utils import MyContainer


class MemoryEnvironment(EnivironmentBase):

    def __init__(self, plot_len=10, time_lag=5, seed=5):

        self._state = MyContainer()

        self._state.plot_len = plot_len
        self._state.time_lag = time_lag
        self._state.action_dim = 2
        self._state.legal_moves = np.arange(self._state.action_dim)
        self._state.obs_dim = plot_len
        self._state.rng = np.random.RandomState(seed)

    @property
    def action_dim(self):
        return self._state.action_dim

    @property
    def obs_dim(self):
        return self._state.obs_dim

    def reset(self):
        self._final_reward = -5
        self._plot = np.zeros(self._state.plot_len)
        self._color = self._state.rng.randint(2)
        self._choices = self._state.rng.choice(self._state.plot_len-2, 2, replace=False)
        self._agent_loc = self._choices[0] + 1
        self._signal_loc = self._choices[1] + 1
        self._plot[self._agent_loc] = 1
        if self._color == 0:
            self._plot[self._signal_loc] = 2
        else:
            self._plot[self._signal_loc] = 3
        self._time = 1
        self._min = 1
        self._max = self._state.plot_len - 2
        self._once_touch = False
        return self._plot, self._state.legal_moves

    def step(self, action):

        self._time += 1
        if self._time == 2:
            self._plot[self._signal_loc] = 0

        if self._time == self._state.time_lag + 2:
            self._tchoices = self._state.rng.randint(2)
            if self._tchoices == 0:
                self._loc4 = 0
                self._loc5 = self._state.plot_len - 1
            else:
                self._loc5 = 0
                self._loc4 = self._state.plot_len - 1
            self._plot[self._loc4] = 4
            self._plot[self._loc5] = 5

        if self._time == self._state.time_lag + 3:
            self._min -= 1
            self._max += 1

        if action == 0:
            if self._agent_loc > self._min:
                self._agent_loc_new = self._agent_loc - 1
            else:
                self._agent_loc_new = self._agent_loc
        else:
            if self._agent_loc < self._max:
                self._agent_loc_new = self._agent_loc + 1
            else:
                self._agent_loc_new = self._agent_loc

        self._plot[self._agent_loc] = 0
        self._plot[self._agent_loc_new] = 1
        self._agent_loc = self._agent_loc_new

        if self._once_touch == False and self._time > (self._state.time_lag + 2):
            if self._agent_loc == self._loc4 or self._agent_loc == self._loc5:
                if self._color == 0 and self._agent_loc == self._loc4:
                    self._final_reward = 5
                elif self._color == 1 and self._agent_loc == self._loc5:
                    self._final_reward = 5

                if self._plot[self._loc4] == 4:
                    self._plot[self._loc4] = 0

                if self._plot[self._loc5] == 5:
                    self._plot[self._loc5] = 0
                self._once_touch = True

        done = False
        if self._time == (self._state.plot_len + self._state.time_lag):
            done = True

        reward = 0
        if done:
            reward = self._final_reward

        return self._plot, self._state.legal_moves, reward, done

    def render(self, mode='rgb_array'):
        pass

    def seed(self, seed):
        self._env.seed(seed=seed)

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
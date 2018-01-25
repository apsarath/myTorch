# Adapted from https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py

import numpy as np
from multiprocessing import Process, Pipe
from myTorch.environment import EnivironmentBase
from myTorch.environment import make_environment

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            obs, legal_moves, reward, done = env.step(data["action"])
            if done:
                if "game_level" in data:
                    obs, legal_moves = env.reset(game_level=data["game_level"])
                else:
                    obs, legal_moves = env.reset()
            remote.send((obs, legal_moves, reward, done))
        elif cmd == 'reset':
            if "game_level" in data:
                obs, legal_moves = env.reset(game_level=data["game_level"])
            else:
                obs, legal_moves = env.reset()
            remote.send((obs, legal_moves))
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.action_dim, env.obs_dim))
        elif cmd == 'get_max_episode_len':
            max_episode_len = env.max_episode_len if hasattr(env, 'max_episode_len') else None
            remote.send(max_episode_len)
        elif cmd == 'have_games_exhausted':
            have_games_exhausted = env.have_games_exhausted if hasattr(env, 'have_games_exhausted') else None
            remote.send(have_games_exhausted)
        else:
            raise NotImplementedError


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class SubprocVecEnv(EnivironmentBase):
    def __init__(self, env_fns):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.closed = False
        self.env_dim = {}
        nenvs = len(env_fns)
        print "Preparing {} environments".format(nenvs)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        self.env_dim["action_dim"], self.env_dim["obs_dim"] = self.remotes[0].recv()

        self.remotes[0].send(('get_max_episode_len', None))
        self._max_episode_len = self.remotes[0].recv()


    def step(self, actions, **kwargs):
        data_to_worker = {"action":None}
        for k in kwargs:
            data_to_worker[k] = kwargs[k]
        for remote, action in zip(self.remotes, actions):
            data_to_worker["action"] = action
            remote.send(('step', data_to_worker))
        results = [remote.recv() for remote in self.remotes]
        obs, legal_moves, rewards, dones = zip(*results)
        return np.stack(obs), np.stack(legal_moves), np.stack(rewards), np.stack(dones)

    def reset(self, **kwargs):
        for remote in self.remotes:
            remote.send(('reset', kwargs))
        results = [remote.recv() for remote in self.remotes]
        obs, legal_moves = zip(*results)
        return np.stack(obs), np.stack(legal_moves)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return

        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def have_games_exhausted(self):
        self.remotes[0].send(('have_games_exhausted', None))
        return self.remotes[0].recv()

    @property
    def num_envs(self):
        return len(self.remotes)

    @property
    def max_episode_len(self):
        return self._max_episode_len

    def render(self, mode='rgb_array'):
        return

    def seed(self, seed):
        return

    def get_random_state(self):
        return

    def set_random_state(self, state):
        return

    def save(self, save_dir):
        pass

    def load(self, save_dir):
        pass
    
    @property
    def action_dim(self):
        return self.env_dim["action_dim"]

    @property
    def obs_dim(self):
        return self.env_dim["obs_dim"]

def get_batched_env(env_name, batch_size=5, seed=1234, game_dir=None, mode=None, is_one_hot_world=False):
    def make_env_fn_wrapper(rank, seed):
        def _thunk():
            env = make_environment(env_name, game_dir, mode, is_one_hot_world)
            env.seed(seed + rank)
            return env
        return _thunk


    env = SubprocVecEnv([make_env_fn_wrapper(i, seed) for i in range(batch_size)])
    return env

if __name__=="__main__":
    env_name = "CartPole-v0"
    batch_size = 3
    env = get_batched_env(env_name, batch_size)
    obs, legal_moves = env.reset()
    assert(obs.shape[0] == legal_moves.shape[0])
    assert(obs[0,:].shape == env.obs_dim)

    obs, legal_moves, rewards, dones = env.step([0]*batch_size)
    assert(obs.shape[0] == legal_moves.shape[0] == rewards.shape[0] == dones.shape[0])
    assert(obs[0,:].shape == env.obs_dim)
    print "Basic tests passed for batched enviroment!"
    env.close()

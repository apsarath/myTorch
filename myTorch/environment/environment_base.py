from abc import ABCMeta, abstractmethod, abstractproperty

class EnvironmentBase():
    __metaclass__ = ABCMeta

    @abstractmethod
    def reset(self):
    	pass

    @abstractmethod
    def step(self, action):
    	pass

    @abstractmethod
    def render(self, mode='rgb_array'):
        pass

    @abstractmethod
    def seed(self, seed):
        pass

    @abstractmethod
    def get_random_state(self):
        pass

    @abstractmethod
    def set_random_state(self, state):
        pass

    @abstractmethod
    def save(self, save_dir):
    	pass

    @abstractmethod
    def load(self, save_dir):
    	pass	

    @abstractproperty
    def action_dim(self):
    	pass

    @abstractproperty
    def obs_dim(self):
    	pass

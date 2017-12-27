from abc import ABCMeta, abstractmethod, abstractproperty

class EnivironmentBase():
    __metaclass__ = ABCMeta

    @abstractmethod
    def reset(self):
    	pass

    @abstractmethod
    def step(self, action):
    	pass

    @abstractmethod
    def save(self, save_dir):
    	pass

    @abstractmethod
    def resume(self, save_dir):
    	pass	

    @abstractproperty
    def action_dim(self):
    	pass

    @abstractproperty
    def obs_dim(self):
    	pass

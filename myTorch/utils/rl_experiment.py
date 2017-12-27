import cPickle as pickle
from myTorch.utils import create_folder

class RLExperiment(object):

	def __init__(self, name, dir_name):

		self._name = name
		self._dir_name = dir_name
		create_folder(self._dir_name)

		self._agent = None
		self._trainer = None
		self._optimizer = None
		self._config = None
		self._replay_buffer = None
		self._logger = None
		self._env = None

	def register_agent(self, agent):
		self._agent = agent

	def register_trainer(self, trainer):
		self._trainer = trainer

	def register_optimizer(self, optimizer):
		self._optimizer = optimizer

	def register_config(self, config):
		self._config = config

	def register_replay_buffer(self, replay_buffer):
		self._replay_buffer = replay_buffer

	def register_logger(self, logger):
		self._logger = logger

	def register_env(self, env):
		self._env = env

	def save(self):
		return

	def is_resumable(self):
		return

	def resume(self):
		return



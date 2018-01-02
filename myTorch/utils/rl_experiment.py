import os
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

	def save(self, tag):		

		print("saving model ...")

		savedir = os.path.join(self._dir_name, tag)
		create_folder(savedir)

		if self._trainer is not None:
			fname = os.path.join(self._dir_name, tag, "trainer.p")
			self._trainer.save(fname)

		if self._config is not None:
			fname = os.path.join(self._dir_name, tag, "config.p")
			self._config.save(fname)

		if self._replay_buffer is not None:
			buffer_dir = os.path.join(self._dir_name, tag, "buffer")
			self._replay_buffer.save(buffer_dir)

	def is_resumable(self):
		return False

	def resume(self, tag):

		print("loading model ...")

		if self._trainer is not None:
			fname = os.path.join(self._dir_name, tag, "trainer.p")
			self._trainer.load(fname)

		if self._config is not None:
			fname = os.path.join(self._dir_name, tag, "config.p")
			self._config.load(fname)
		
		if self._replay_buffer is not None:
			buffer_dir = os.path.join(self._dir_name, tag, "buffer")
			self._replay_buffer.load(buffer_dir)



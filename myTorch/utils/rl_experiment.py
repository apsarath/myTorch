import os
from os.path import isfile
import cPickle as pickle
from shutil import rmtree

import myTorch
from myTorch.utils import create_folder


class RLExperiment(object):

	def __init__(self, name, dir_name, backup_logger=False):

		self._name = name
		self._dir_name = dir_name
		create_folder(self._dir_name)
		self._backup_logger = backup_logger

		self._agent = None
		self._trainer = None
		self._config = None
		self._replay_buffer = None
		self._logger = None
		self._env = None

	def register_agent(self, agent):
		self._agent = agent

	def register_trainer(self, trainer):
		self._trainer = trainer

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

		flagfile = os.path.join(savedir, "flag.p")
		if isfile(flagfile):
			os.remove(flagfile)

		if self._agent is not None:
			self._agent.save(savedir)

		if self._trainer is not None:
			fname = os.path.join(self._dir_name, tag, "trainer.p")
			self._trainer.save(fname)

		if self._config is not None:
			fname = os.path.join(self._dir_name, tag, "config.p")
			self._config.save(fname)

		if self._replay_buffer is not None:
			buffer_dir = os.path.join(self._dir_name, tag, "buffer")
			self._replay_buffer.save(buffer_dir)

		if self._env is not None:
			fname = os.path.join(self._dir_name, tag, "env.p")
			self._env.save(fname)

		if self._logger is not None:
			if self._backup_logger:
				self._logger.save(os.path.join(self._dir_name,tag,"logger"))

		f = open(flagfile, "w")
		f.close()


	def is_resumable(self, tag):

		flagfile = os.path.join(self._dir_name, tag, "flag.p")
		if isfile(flagfile):
			return True
		else:
			return False

	def resume(self, tag):

		print("loading model ...")

		savedir = os.path.join(self._dir_name, tag)

		if self._agent is not None:
			self._agent.load(savedir)

		if self._trainer is not None:
			fname = os.path.join(self._dir_name, tag, "trainer.p")
			self._trainer.load(fname)

		if self._config is not None:
			fname = os.path.join(self._dir_name, tag, "config.p")
			self._config.load(fname)
		
		if self._replay_buffer is not None:
			buffer_dir = os.path.join(self._dir_name, tag, "buffer")
			self._replay_buffer.load(buffer_dir)

		if self._env is not None:
			fname = os.path.join(self._dir_name, tag, "env.p")
			self._env.load(fname)


		if self._logger is not None:
			if self._backup_logger:
				self._logger.load(os.path.join(self._dir_name,tag,"logger"))

	def force_restart(self, tag):

		print("force restarting...")

		savedir = os.path.join(self._dir_name, tag)
		create_folder(savedir)
		rmtree(savedir)

		if self._logger is not None:
			self._logger.force_restart()





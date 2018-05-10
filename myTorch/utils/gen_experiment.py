import os
from os.path import isfile
from shutil import rmtree

import myTorch
from myTorch.utils import create_folder


class GenExperiment(object):

	def __init__(self, name, dir_name, backup_logger=False):

		self._name = name
		self._dir_name = dir_name
		create_folder(self._dir_name)
		self._backup_logger = backup_logger
		self._data = {}

	def register(self, tag, obj):
		self._data[tag] = obj

	def save(self, tag, input_obj_tag=None):		

		print("saving model ...")

		savedir = os.path.join(self._dir_name, tag)
		create_folder(savedir)

		flagfile = os.path.join(savedir, "flag.p")
		if isfile(flagfile):
			os.remove(flagfile)

		if input_obj_tag is None:
			for obj_tag, obj in list(self._data.items()):
				folder_name = os.path.join(savedir, obj_tag)
				create_folder(folder_name)
				obj.save(folder_name)
		else:
			folder_name = os.path.join(savedir, input_obj_tag)
			create_folder(folder_name)
			self._data[input_obj_tag].save(folder_name)

		f = open(flagfile, "w")
		f.close()

	def is_resumable(self, tag):

		flagfile = os.path.join(self._dir_name, tag, "flag.p")
		if isfile(flagfile):
			return True
		else:
			return False

	def resume(self, tag, input_obj_tag=None):

		print("loading model ...")
		savedir = os.path.join(self._dir_name, tag)
		if input_obj_tag is None:
			for obj_tag, obj in list(self._data.items()):
				folder_name = os.path.join(savedir, obj_tag)
				obj.load(folder_name)
		else:
			folder_name = os.path.join(savedir, input_obj_tag)
			self._data[input_obj_tag].load(folder_name)

	def force_restart(self, tag):

		print("force restarting...")

		savedir = os.path.join(self._dir_name, tag)
		create_folder(savedir)
		rmtree(savedir)

		if self._data["logger"] is not None:
			self._data["logger"].force_restart()





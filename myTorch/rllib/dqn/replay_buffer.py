import os
import numpy as np
import cPickle as pickle

import myTorch
from myTorch.utils import create_folder

class ReplayBuffer(object):

	def __init__(self, obs_dim, action_dim, numpy_rng, size=1e5, compress=False):

		self._obs_dim = int(obs_dim)
		self._action_dim = int(action_dim)
		self._numpy_rng = numpy_rng
		self._size = int(size)
		self._compress = compress

		dtype = "float32"
		if self._compress == True:
			dtype = "int8"

		self._data = {}
		self._data["observations"] = np.zeros(shape=[self._size, self._obs_dim], dtype=dtype)
		self._data["legal_moves"] = np.zeros(shape=[self._size, self._action_dim], dtype=dtype)
		self._data["actions"] = np.zeros(shape=[self._size, self._action_dim], dtype=dtype)
		self._data["rewards"] = np.zeros(shape=[self._size], dtype=dtype)
		self._data["observations_tp1"] = np.zeros(shape=[self._size, self._obs_dim], dtype=dtype)
		self._data["legal_moves_tp1"] = np.zeros(shape=[self._size, self._action_dim], dtype=dtype)
		self._data["pcontinues"] = np.zeros_like(self._data["rewards"], dtype=dtype)

		self._write_index = -1
		self._n = 0


	def add(self, data):

		self._write_index = (self._write_index + 1) % self._size
		self._n = int(min(self._size, self._n + 1))
		for key in self._data:
			self._data[key][self._write_index] = data[key]

	def sample_minibatch(self, batch_size=32):

		if self._n < batch_size:
			raise IndexError("Buffer does not have batch_size=%d transitions yet." % batch_size)

		indices = self._numpy_rng.choice(self._n, size=batch_size, replace=False)
		rval = {}
		for key in self._data:
			rval[key] = self._data[key][indices]

		if self._compress == True:
			for key in rval:
				rval[key] = np.asarray(rval[key], dtype="float32")

		return rval

	def save(self, fname):

		create_folder(fname)

		sdict = {}
		sdict["obs_dim"] = self._obs_dim
		sdict["action_dim"] = self._action_dim
		sdict["size"] = self._size
		sdict["write_index"] = self._write_index
		sdict["n"] = self._n

		
		full_name = os.path.join(fname, "meta.ckpt")
		with open(full_name, "wb") as f:
			pickle.dump(sdict, f)

		for key in self._data:
			full_name = os.path.join(fname, "{}.npy".format(key))
			with open(full_name,"w") as f:
				np.save(f, self._data[key])

	def load(self, fname):

		full_name = os.path.join(fname, "meta.ckpt")
		with open(full_name, "rb") as f:
			sdict = pickle.load(f)

		self._obs_dim = sdict["obs_dim"]
		self._action_dim = sdict["action_dim"]
		self._size = sdict["size"]
		self._write_index = sdict["write_index"]
		self._n = sdict["n"]

		for key in self._data:
			full_name = os.path.join(fname, "{}.npy".format(key))
			with open(full_name,"r") as f:
				self._data[key] = np.load(f) 




		

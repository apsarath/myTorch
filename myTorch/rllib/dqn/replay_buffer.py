import os
import numpy as np
import _pickle as pickle

import myTorch
from myTorch.utils import create_folder

class ReplayBuffer(object):

    def __init__(self, numpy_rng, size=1e5, compress=False):

        self._numpy_rng = numpy_rng
        self._size = int(size)
        self._compress = compress

        self._data_keys = ["observations", "legal_moves", "actions", "rewards", "observations_tp1", "legal_moves_tp1", "pcontinues"]
        self._dtype = {}
        for key in self._data_keys:
            self._dtype[key] = "int8"

        if not self._compress:
            for key in ["observations", "rewards", "observations_tp1"]:
                self._dtype[key] = "float32"

        self._data = {}
        for data_key in self._data_keys:
            self._data[data_key] = [None]*int(size)

        self._write_index = -1
        self._n = 0


    def add(self, data):

        self._write_index = (self._write_index + 1) % self._size
        self._n = int(min(self._size, self._n + 1))
        for key in self._data:
            self._data[key][self._write_index] = np.asarray(data[key], dtype=self._dtype[key])

    def sample_minibatch(self, batch_size=32):

        if self._n < batch_size:
            raise IndexError("Buffer does not have batch_size=%d transitions yet." % batch_size)

        indices = self._numpy_rng.choice(self._n, size=batch_size, replace=False)
        rval = {}
        for key in self._data:
            rval[key] = np.asarray([self._data[key][idx] for idx in indices], dtype="float32")

        return rval

    def save(self, fname):

        create_folder(fname)

        sdict = {}
        sdict["size"] = self._size
        sdict["write_index"] = self._write_index
        sdict["n"] = self._n

        
        full_name = os.path.join(fname, "meta.ckpt")
        with open(full_name, "wb") as f:
            pickle.dump(sdict, f)

        for key in self._data:
            full_name = os.path.join(fname, "{}.npy".format(key))
            with open(full_name,"wb") as f:
                np.save(f, self._data[key])

    def load(self, fname):

        full_name = os.path.join(fname, "meta.ckpt")
        with open(full_name, "rb") as f:
            sdict = pickle.load(f)

        self._size = sdict["size"]
        self._write_index = sdict["write_index"]
        self._n = sdict["n"]

        for key in self._data:
            full_name = os.path.join(fname, "{}.npy".format(key))
            with open(full_name,"rb") as f:
                self._data[key] = np.load(f) 

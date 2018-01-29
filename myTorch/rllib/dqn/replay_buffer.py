import os
import numpy as np
import pickle

import myTorch
from myTorch.utils import create_folder

class ReplayBuffer(object):

    def __init__(self, numpy_rng, size=1e5, compress=False):

        self._numpy_rng = numpy_rng
        self._size = int(size)
        self._compress = compress

        self._dtype = "float32"
        if self._compress == True:
            self._dtype = "int8"

        self._data_types = ["observations", "legal_moves", "actions", "rewards", "observations_tp1", "legal_moves_tp1", "pcontinues"]
        self._data = {}
        for data_type in self._data_types:
            self._data[data_type] = [None]*int(size)

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
            rval[key] = np.asarray([self._data[key][idx] for idx in indices], dtype=self._dtype)
    
        if self._compress == True:
            for key in rval:
                rval[key] = np.asarray(rval[key], dtype="float32")

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
            with open(full_name,"w") as f:
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
            with open(full_name,"r") as f:
                self._data[key] = np.load(f) 




        

"""Copy Task."""
import numpy as np
import _pickle as pickle

class CopyData(object):
    """Copy task data generator."""

    def __init__(self, num_bits=8, min_len=1, max_len=20, batch_size=5):
        
        self._num_bits = num_bits
        self._min_len = min_len
        self._max_len = max_len
        self._batch_size = batch_size
        self._examples_seen = 0
        self._rng = np.random.RandomState(5)

    def next(self, seq_len=None, batch_size=None):
        
        if seq_len is None:
            seq_len = self._rng.random_integers(self._min_len, self._max_len)
        if batch_size is None:
            batch_size = self._batch_size

        x = np.zeros((2*seq_len+1, batch_size, self._num_bits), dtype="float32")
        y = np.zeros((2*seq_len+1, batch_size, self._num_bits), dtype="float32")
        mask = np.zeros(2*seq_len+1, dtype="float32")
        data = self._rng.binomial(1, 0.5, size=(seq_len, batch_size, self._num_bits))
        data[:, :, -1] = 0
        x[0:seq_len] = data
        x[seq_len, :, -1] = 1
        y[seq_len+1:] = data
        mask[seq_len+1:] = 1
        output = {}
        output['x'] = x
        output['y'] = y
        output['mask'] = mask
        output['seqlen'] = seq_len
        output['datalen'] = 2*seq_len+1
        self._examples_seen += batch_size
        return output

    def save(self, file_name):

        state = {}
        state["num_bits"] = self._num_bits
        state["min_len"] = self._min_len
        state["max_len"] = self._max_len
        state["batch_size"] = self._batch_size
        state["rng_state"] = self._rng.get_state()
        state["examples_seen"] = self._examples_seen
        pickle.dump(state, open(file_name, "wb"))

    def load(self, file_name):

        state = pickle.load(open(file_name, "rb"))
        self._num_bits = state["num_bits"]
        self._min_len = state["min_len"]
        self._max_len = state["max_len"]
        self._batch_size = state["batch_size"]
        self._batch_size = state["batch_size"]
        self._rng.set_state(state["rng_state"])
        self._examples_seen = state["examples_seen"]


if __name__=="__main__":

    gen = CopyData()
    data = gen.next(seq_len=2)
    print(data['seqlen'])
    print(data['x'])
    print(data['y'])
    print(data['mask'])



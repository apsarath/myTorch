"""Copy Task."""
import numpy as np
from myTorch.utils import MyContainer


class CopyData(object):
    """Copy task data generator."""

    def __init__(self, num_bits=8, min_len=1, max_len=20, batch_size=5, seed=5):
        """Initializes the data generator.

        Args:
            num_bits: int, number of bits in the vector.
            min_len: int, minimum length of the sequence.
            max_len int, maximum length of the sequence.
            batch_size: int, batch size.
            seed: int, random seed.
        """

        self._state = MyContainer()

        self._state.num_bits = num_bits
        self._state.min_len = min_len
        self._state.max_len = max_len
        self._state.batch_size = batch_size
        self._state.examples_seen = 0
        self._state.rng = np.random.RandomState(seed)

    def next(self, seq_len=None, batch_size=None):
        """Returns next batch of data.

        Args:
            seq_len: int, length of the sequence.
            batch_size: int, batch size.
        """

        if seq_len is None:
            seq_len = self._state.rng.random_integers(self._state.min_len, self._state.max_len)
        if batch_size is None:
            batch_size = self._state.batch_size

        x = np.zeros((2*(seq_len+1), batch_size, self._state.num_bits), dtype="float32")
        y = np.zeros((2*(seq_len+1), batch_size, self._state.num_bits), dtype="float32")
        mask = np.zeros(2*(seq_len+1), dtype="float32")

        data = self._state.rng.binomial(1, 0.5, size=(seq_len+1, batch_size, self._state.num_bits))
        data_random = self._state.rng.binomial(1, 0.5, size=(seq_len+1, batch_size, self._state.num_bits))

        data[:, :, -1] = 0
        data[seq_len, :, :] = 0
        data[seq_len, :, -1] = 1

        data_random[:, :, -1] = 0
        data_random[seq_len, :, :] = 0
        data_random[seq_len, :, -1] = 1

        x[0:seq_len+1] = data
        y[seq_len+1:] = data_random
        mask[seq_len+1:] = 1

        output = {}
        output['x'] = x
        output['y'] = y
        output['mask'] = mask
        output['seqlen'] = seq_len
        output['datalen'] = 2*(seq_len+1)
        self._state.examples_seen += batch_size
        return output

    def save(self, file_name):
        """Saves the state of the data generator.

        Args:
            file_name: str, file name with absolute path.
        """

        self._state.save(file_name)

    def load(self, file_name):
        """Loads the state of the data generator.

        Args:
            file_name: str, file name with absolute path.
        """

        self._state.load(file_name)


if __name__=="__main__":

    gen = CopyData()
    data = gen.next(seq_len=2)
    print(data['seqlen'])
    print(data['x'])
    print(data['y'])
    print(data['mask'])



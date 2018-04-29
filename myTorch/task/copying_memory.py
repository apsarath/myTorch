"""Copying Memory Task."""
import numpy as np
from myTorch.utils import MyContainer


class CopyingMemoryData(object):
    """Copying Memory task data generator.
    As defined in https://arxiv.org/pdf/1511.06464.pdf
    """

    def __init__(self, seq_len=10, time_lag=10, batch_size=5, seed=5):
        """Initializes the data generator.

        Args:
            seq_len: int, length of data sequence.
            time_lag: int, number of time steps lag from input to output.
            batch_size: int, batch size.
            seed: int, random seed.
        """

        self._state = MyContainer()

        self._state.seq_len = seq_len
        self._state.time_lag = time_lag
        self._state.batch_size = batch_size
        self._state.examples_seen = 0
        self._state.rng = np.random.RandomState(seed)

    def next(self, batch_size=None):
        """Returns next batch of data.

        Args:
            batch_size: int, batch size.
        """

        if batch_size is None:
            batch_size = self._state.batch_size

        data_len = 2 * self._state.seq_len + self._state.time_lag

        x = np.zeros((data_len, batch_size, 1), dtype="float32")
        y = np.zeros((data_len, batch_size, 1), dtype="float32")
        mask = np.zeros(data_len, dtype="float32")

        data = self._state.rng.randint(1, high=9, size=(self._state.seq_len, batch_size, 1))

        x[0:self._state.seq_len] = data
        x[self._state.seq_len+self._state.time_lag-1] = 9
        y[self._state.seq_len + self._state.time_lag:] = data
        mask[self._state.seq_len + self._state.time_lag:] = 1

        output = {}
        output['x'] = x
        output['y'] = y
        output['mask'] = mask
        output['seqlen'] = self._state.seq_len
        output['datalen'] = data_len
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

    gen = CopyingMemoryData()
    data = gen.next()
    print(data['seqlen'])
    print(data['x'])
    print(data['y'])
    print(data['mask'])



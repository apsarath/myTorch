"""Denoising Task."""
import numpy as np
from myTorch.utils import MyContainer, one_hot


class DenoisingData(object):
    """Denoising task data generator.
    As defined in https://arxiv.org/pdf/1706.02761.pdf
    """

    def __init__(self, seq_len=10, time_lag=15, batch_size=5, seed=5):
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

        data_len = self._state.seq_len + self._state.time_lag + 1

        mask = np.zeros(data_len, dtype="float32")
        mask[-self._state.seq_len:] = 1.0

        seq = np.random.randint(1, high=9, size=(batch_size, self._state.seq_len))
        zeros1 = np.zeros((batch_size, self._state.time_lag))

        for i in range(batch_size):
            ind = np.random.choice(self._state.time_lag, self._state.seq_len, replace=False)
            ind.sort()
            zeros1[i][ind] = seq[i]

        zeros2 = np.zeros((batch_size, self._state.time_lag + 1))
        marker = 9 * np.ones((batch_size, 1))
        zeros3 = np.zeros((batch_size, self._state.seq_len))

        x = np.concatenate((zeros1, marker, zeros3), axis=1).astype('int32')
        y = np.concatenate((zeros2, seq), axis=1).astype('int64')

        x = np.expand_dims(x.T, -1)
        y = np.expand_dims(y.T, -1)
    
        one_hot_x = np.zeros((x.shape[0], x.shape[1], 10), dtype="float32")
        for i in range(data_len):
            one_hot_x[i] = one_hot(x[i], 10).astype(np.float32)

        output = {}
        output['x'] = one_hot_x
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

    gen = DenoisingData()
    data = gen.next()
    print(data['seqlen'])
    print(data['x'])
    print(data['y'])
    print(data['mask'])



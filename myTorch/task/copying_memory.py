"""Copying Memory Task."""
import numpy as np
from myTorch.utils import MyContainer, one_hot


class CopyingMemoryData(object):
    """Copying Memory task data generator.
    As defined in https://arxiv.org/pdf/1511.06464.pdf
    """

    def __init__(self, seq_len=10, time_lag_min=100, time_lag_max=100, num_digits=8, num_noise_digits=1, batch_size=5, seed=5):
        """Initializes the data generator.

        Args:
            seq_len: int, length of data sequence.
            time_lag_min: int, Min number of time steps lag from input to output.
            time_lag_max: int, Max number of time steps lag from input to output.
            num_digits: Range of the numbers to sample the values of the "output sequence"
            num_noise_digits: 
            batch_size: int, batch size.
            seed: int, random seed.
        """

        self._state = MyContainer()

        self._state.seq_len = seq_len
        self._state.time_lag_range = [time_lag_min, time_lag_max]
        self._state.num_digits = num_digits
        self._state.num_noise_digits = num_noise_digits
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

        time_lag = self._state.rng.randint(self._state.time_lag_range[0], self._state.time_lag_range[1] + 1)

        data_len = 2 * self._state.seq_len + time_lag

        digit_range = self._state.num_noise_digits + self._state.num_digits + 1
        noise_id_range = [0, self._state.num_noise_digits-1]
        seq_id_range = [self._state.num_noise_digits, digit_range - 2]
        marker_id = digit_range - 1

        x = self._state.rng.randint(noise_id_range[0], high=noise_id_range[1]+1, size=(data_len, batch_size, 1))
        one_hot_x = np.zeros((data_len, batch_size, digit_range), dtype="float32")
        y = np.zeros((data_len, batch_size, 1), dtype="int64")
        mask = np.ones(data_len, dtype="float32")

        data = self._state.rng.randint(seq_id_range[0], high=seq_id_range[1]+1, size=(self._state.seq_len, batch_size, 1))

        x[0: self._state.seq_len] = data
        x[self._state.seq_len + time_lag - 1] = marker_id
        y[self._state.seq_len + time_lag:] = data
        for i in range(data_len):
            one_hot_x[i] = one_hot(x[i], digit_range).astype(np.float32)

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

    gen = CopyingMemoryData()
    data = gen.next()
    print(data['seqlen'])
    print(data['x'])
    print(data['y'])
    print(data['mask'])



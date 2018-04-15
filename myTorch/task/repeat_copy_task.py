"""Repeat Copy Task."""
import numpy as np
import _pickle as pickle
from myTorch.utils import MyContainer


class RepeatCopyData(object):
    """Repeat Copy task data generator."""

    def __init__(self, num_bits=8, min_len=1, max_len=10, min_repeat=1, max_repeat=10, batch_size=5, seed=5):
        """Initializes the data generator.

        Args:
            num_bits: int, number of bits in the vector.
            min_len: int, minimum length of the sequence.
            max_len int, maximum length of the sequence.
            min_repeat: int, minimum number of times to repeat.
            max_repeat: int, maximum number of times to repeat.
            batch_size: int, batch size.
            seed: int, random seed.
        """

        self._state = MyContainer()

        self._state.num_bits = num_bits
        self._state.min_len = min_len
        self._state.max_len = max_len
        self._state.min_repeat = min_repeat
        self._state.max_repeat = max_repeat
        self._state.batch_size = batch_size
        self._state.examples_seen = 0
        self._state.rng = np.random.RandomState(seed)

        self._initialize_normalizer()

    def _initialize_normalizer(self):
        """Initializes the normalizer for repeat_num."""

        self._state.reps_mean = (self._state.max_repeat + self._state.min_repeat) / 2

        reps_var = (((self._state.max_repeat - self._state.min_repeat + 1) ** 2) - 1) / 12
        self._state.reps_std = np.sqrt(reps_var)

    def _normalize_repeat(self, reps):
        """Normalizes repeat_num.

        Args:
            reps: int, number of times to repeat.
        """

        return (reps - self._state.reps_mean) / self._state.reps_std

    def next(self, seq_len=None, num_repeat=None, batch_size=None):
        """Returns next batch of data.

        Args:
            seq_len: int, length of the sequence.
            num_repeat: int, number of times to repeat.
            batch_size: int, batch size.
        """

        if seq_len is None:
            seq_len = self._state.rng.random_integers(self._state.min_len, self._state.max_len)

        if num_repeat is None:
            num_repeat = self._state.rng.random_integers(self._state.min_repeat, self._state.max_repeat)

        if batch_size is None:
            batch_size = self._state.batch_size

        x = np.zeros(((seq_len + 1)*(num_repeat + 1) + 1, batch_size, self._state.num_bits), dtype="float32")
        y = np.zeros(((seq_len + 1)*(num_repeat + 1) + 1, batch_size, self._state.num_bits), dtype="float32")
        mask = np.zeros((seq_len + 1)*(num_repeat + 1) + 1, dtype="float32")

        data = self._state.rng.binomial(1, 0.5, size=(seq_len+1, batch_size, self._state.num_bits))
        data[:, :, -1] = 0
        data[seq_len, :, :] = 0
        data[seq_len, :, -1] = 1
        repeated_data = np.concatenate([data for _ in range(num_repeat)])

        x[0:seq_len+1] = data
        x[seq_len+1, :, :] = self._normalize_repeat(num_repeat)
        y[seq_len+2:] = repeated_data

        mask[seq_len + 2:] = 1

        output = {}
        output['x'] = x
        output['y'] = y
        output['mask'] = mask
        output['seqlen'] = seq_len
        output['datalen'] = (seq_len + 1)*(num_repeat + 1) + 1
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
        self._initialize_normalizer()


if __name__ == "__main__":
    gen = RepeatCopyData()
    data = gen.next(seq_len=2)
    print(data['seqlen'])
    print(data['x'])
    print(data['y'])
    print(data['mask'])



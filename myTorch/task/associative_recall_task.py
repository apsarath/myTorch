"""Copy Task."""
import numpy as np
from myTorch.utils import MyContainer


class AssociativeRecallData(object):
    """Associative Recall task data generator."""

    def __init__(self, num_bits=8, min_len=2, max_len=6, block_len=3, batch_size=5, seed=5):
        """Initializes the data generator.

        Args:
            num_bits: int, number of bits in the vector.
            min_len: int, minimum length of the sequence.
            max_len int, maximum length of the sequence.
            block_len: int, length of the block.
            batch_size: int, batch size.
            seed: int, random seed.
        """

        self._state = MyContainer()

        self._state.num_bits = num_bits
        self._state.min_len = min_len
        self._state.max_len = max_len
        self._state.block_len = block_len
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

        stride = self._state.block_len + 1

        x = np.zeros(((stride*(seq_len+2))+1, batch_size, self._state.num_bits), dtype="float32")
        y = np.zeros(((stride*(seq_len+2))+1, batch_size, self._state.num_bits), dtype="float32")
        mask = np.zeros((stride*(seq_len+2))+1, dtype="float32")

        data = self._state.rng.binomial(1, 0.5, size=(stride*seq_len, batch_size,
                                                      self._state.num_bits))
        data[:, :, -2:] = 0
        for i in range(1, seq_len+1):
            data[i*stride - 1, :, :] = 0
            data[i*stride - 1, :, -2] = 1

        x[0: stride * seq_len] = data
        x[stride * seq_len, :, :] = 0
        x[stride * seq_len, :, -1] = 1

        key = self._state.rng.randint(0, seq_len-1)
        query = x[key*stride: (key+1)*stride - 1]
        x[stride * seq_len + 1: stride * (seq_len + 1)] = query
        x[stride * (seq_len+1), :, -1] = 1

        result = x[(key+1)*stride: (key+2)*stride - 1]
        y[stride * (seq_len+1) + 1: stride * (seq_len+2)] = result
        y[-1, :, -1] = 1

        mask[stride * (seq_len + 1) + 1:] = 1

        output = {}
        output['x'] = x
        output['y'] = y
        output['mask'] = mask
        output['seqlen'] = seq_len
        output['datalen'] = (stride*(seq_len+2))+1
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

    gen = AssociativeRecallData()
    data = gen.next(seq_len=4)
    print(data['seqlen'])
    print(data['x'])
    print(data['y'])
    print(data['mask'])



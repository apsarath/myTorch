"""Adding Task."""
import numpy as np
from myTorch.utils import MyContainer


class AddingData(object):
    """Adding task data generator.
    As defined in https://arxiv.org/pdf/1511.06464.pdf
    """

    def __init__(self, seq_len=10, batch_size=5, seed=5):
        """Initializes the data generator.

        Args:
            seq_len: int, length of single sequence.
            time_lag: int, number of time steps lag from input to output.
            batch_size: int, batch size.
            seed: int, random seed.
        """

        self._state = MyContainer()

        self._state.seq_len = seq_len
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

        data_len = 2 * self._state.seq_len + 1

        x = np.zeros((data_len, batch_size), dtype="float32")
        y = np.zeros((data_len, batch_size), dtype="float32")
        mask = np.zeros(data_len, dtype="float32")

        data1 = np.random.uniform(low=0., high=1., size=(self._state.seq_len, batch_size))
        data2 = np.zeros((self._state.seq_len, batch_size))
        inds = np.asarray(np.random.randint(self._state.seq_len / 2, size=(batch_size, 2)))
        inds[:, 1] += self._state.seq_len // 2

        for i in range(int(batch_size)):
            data2[inds[i, 0], i] = 1.0
            data2[inds[i, 1], i] = 1.0

        output = (data1 * data2).sum(axis=0)

        x[0:self._state.seq_len] = data1
        x[self._state.seq_len:-1] = data2
        y[-1] = output

        x = np.expand_dims(x, -1)
        y = np.expand_dims(y, -1)

        mask[-1] = 1

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

    gen = AddingData()
    data = gen.next()
    import pdb
    pdb.set_trace()
    print(data['seqlen'])
    print(data['x'])
    print(data['y'])
    print(data['mask'])

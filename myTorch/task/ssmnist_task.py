"""Sequential Stroke MNIST Task."""
import numpy as np
import _pickle as pickle
import os
import math
from myTorch.utils import MyContainer


class SSMNISTData(object):
    """SSMNIST task data generator."""

    def __init__(self, data_folder, num_digits=5, batch_size=10, seed=5):
        """Initializes the data generator.

        Args:
            num_digits: int, number of digits in the sequence.
            batch_size: int, batch size.
            seed: int, random seed.
        """

        self._state = MyContainer()

        self._state.data_folder = data_folder
        self._state.num_digits = num_digits
        self._state.batch_size = int(batch_size)
        self._state.examples_seen = 0
        self._state.rng = np.random.RandomState(seed)

        self._load_data()

        self._reset_iterator()

    def _load_data(self):

        self._data = MyContainer()

        self._data.x = {}
        self._data.y = {}
        self._data.seq_len = {}

        for fold in ["train", "valid", "test"]:
            x_file = os.path.join(self._state.data_folder, str(self._state.num_digits), "x_"+fold+".pkl")
            self._data.x[fold] = pickle.load(open(x_file, "rb"))
            self._data.y[fold] = np.load(os.path.join(self._state.data_folder, str(self._state.num_digits),
                                                      "y_"+fold+".npy"))
            self._data.seq_len[fold] = np.load(os.path.join(self._state.data_folder,
                                                            str(self._state.num_digits), "seq_len_"+fold+".npy"))
            self._data.seq_len[fold] = np.asarray(self._data.seq_len[fold], dtype="int32")

    def _reset_iterator(self):

        self._state.iter_list = {}
        self._state.batches = {}
        self._state.batches_done = {}

        for fold in ["train", "valid", "test"]:
            self._state.iter_list[fold] = self._state.rng.permutation(len(self._data.x[fold]))
            self._state.batches[fold] = math.ceil(len(self._data.x[fold]) // self._state.batch_size)
            self._state.batches_done[fold] = 0

    def next(self, tag):
        """Returns next batch of data.

        Args:
            tag: str, "train" or "valid" or "test"
        """

        if self._state.batches_done[tag] == self._state.batches[tag]:
            return None

        start_ind = self._state.batches_done[tag] * self._state.batch_size
        end_ind = (self._state.batches_done[tag] + 1) * self._state.batch_size
        if end_ind > len(self._data.x[tag]):
            end_ind = len(self._data.x[tag])

        indices = self._state.iter_list[tag][start_ind:end_ind]

        x = [self._data.x[tag][i] for i in indices]
        y = self._data.y[tag][indices]
        seq_len = self._data.seq_len[tag][indices]

        max_len = max(seq_len)

        data_len = int(max_len + 1 + self._state.num_digits + 1)

        new_x = np.zeros((self._state.batch_size, data_len, 5))
        new_y = np.zeros((self._state.batch_size, data_len))
        mask = np.zeros((self._state.batch_size, data_len))

        for i in range(0, len(indices)):
            new_x[i][0:seq_len[i]][:, 0:4] = x[i]
            new_x[i][seq_len[i]][4] = 1
            new_y[i][seq_len[i]+1: seq_len[i]+1+self._state.num_digits+1] = y[i]
            mask[i][seq_len[i]+1: seq_len[i]+1+self._state.num_digits+1] = 1

        new_x = np.swapaxes(new_x, 0, 1)
        new_y = np.swapaxes(new_y, 0, 1).astype('int64')
        mask = np.swapaxes(mask, 0, 1)

        output = {}
        output['x'] = new_x
        output['y'] = new_y
        output['mask'] = mask
        output['datalen'] = data_len

        self._state.batches_done[tag] += 1

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

    gen = SSMNISTData(data_folder="/mnt/data/sarath/data/ssmnist/data/")

    data = None
    count = 0
    while True:
        data = gen.next("valid")
        if data is not None:
            count += 1
            print(count)
        else:
            break
    print(count)

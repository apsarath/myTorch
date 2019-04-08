"""Permuted Sequential MNIST Task."""
import numpy as np
import _pickle as pickle
import os
import math
from myTorch.utils import MyContainer
from myTorch.task.mnist.download_mnist import download_mnist


class PMNISTData(object):
    """Permuted Sequential MNIST task data generator."""

    def __init__(self, batch_size=10, seed=5):
        """Initializes the data generator.

        Args:
            num_digits: int, number of digits in the sequence.
            batch_size: int, batch size.
            seed: int, random seed.
        """

        self._state = MyContainer()

        self._state.batch_size = int(batch_size)
        self._state.examples_seen = 0
        self._state.rng = np.random.RandomState(seed)

        self._load_data()

        self.reset_iterator()

    def _load_data(self):

        data_dir = os.path.join(os.environ["MYTORCH_DATA"], "mnist")
        download_mnist(data_dir)

        self._data = MyContainer()

        self._data.x = {}
        self._data.y = {}

        train_x = np.load(os.path.join(data_dir, "train_x.npy")).astype("float32") / 255
        train_y = np.load(os.path.join(data_dir, "train_y.npy")).astype("float32")

        valid_x = np.load(os.path.join(data_dir, "valid_x.npy")).astype("float32") / 255
        valid_y = np.load(os.path.join(data_dir, "valid_y.npy")).astype("float32")

        test_x = np.load(os.path.join(data_dir, "test_x.npy")).astype("float32") / 255
        test_y = np.load(os.path.join(data_dir, "test_y.npy")).astype("float32")




        train_x = train_x.reshape(train_x.shape[0], train_x.shape[1]*train_x.shape[2])
        valid_x = valid_x.reshape(valid_x.shape[0], valid_x.shape[1]*valid_x.shape[2])
        test_x = test_x.reshape(test_x.shape[0], test_x.shape[1] * test_x.shape[2])

        seq_perm = self._state.rng.permutation(train_x.shape[1])



        self._data.x["valid"] = np.expand_dims(valid_x, 2)
        self._data.y["valid"] = valid_y

        self._data.x["train"] = np.expand_dims(train_x, 2)
        self._data.y["train"] = train_y

        self._data.x["test"] = np.expand_dims(test_x, 2)
        self._data.y["test"] = test_y

        for fold in ["train", "valid", "test"]:
            for i in range(0, len(self._data.x[fold])):
                self._data.x[fold][i] = self._data.x[fold][i][seq_perm]


    def reset_iterator(self):

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

        max_len = len(x[0])

        data_len = int(max_len + 1)

        new_x = np.zeros((self._state.batch_size, data_len, 1))
        new_y = np.zeros((self._state.batch_size, data_len))
        mask = np.zeros((self._state.batch_size, data_len))

        for i in range(0, len(indices)):
            new_x[i][0:max_len] = x[i]
            new_y[i][max_len] = y[i]
            mask[i][max_len] = 1

        new_x = np.swapaxes(new_x, 0, 1).astype('float32')
        new_y = np.swapaxes(new_y, 0, 1).astype('int64')
        mask = np.swapaxes(mask, 0, 1).astype('float32')

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

    gen = PMNISTData()

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

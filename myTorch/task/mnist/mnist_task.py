"""MNIST Task."""
import numpy as np
import os
import math
from myTorch.utils import MyContainer, one_hot
from myTorch.task.mnist.download_mnist import download_mnist


class MNISTData(object):
    """MNIST task data generator."""

    def __init__(self, batch_size=10, seed=5, use_one_hot=True):
        """Initializes the data generator.

        Args:
            batch_size: int, batch size.
            seed: int, random seed.
        """

        self._state = MyContainer()

        self._state.batch_size = int(batch_size)
        self._state.use_one_hot = use_one_hot
        self._state.examples_seen = 0
        self._state.rng = np.random.RandomState(seed)

        self._load_data()

        self.reset_iterator()

    def _load_data(self):
        """Loads the data into memory."""

        data_dir = os.path.join(os.environ["MYTORCH_DATA"], "mnist")
        download_mnist(data_dir)

        self._data = MyContainer()

        self._data.x = {}
        self._data.y = {}

        self._data.x["train"] = np.load(os.path.join(data_dir, "train_x.npy")).astype("float32") / 255
        self._data.y["train"] = np.load(os.path.join(data_dir, "train_y.npy")).astype("float32")

        self._data.x["valid"] = np.load(os.path.join(data_dir, "valid_x.npy")).astype("float32") / 255
        self._data.y["valid"] = np.load(os.path.join(data_dir, "valid_y.npy")).astype("float32")

        self._data.x["test"] = np.load(os.path.join(data_dir, "test_x.npy")).astype("float32") / 255
        self._data.y["test"] = np.load(os.path.join(data_dir, "test_y.npy")).astype("float32")

        if self._state.use_one_hot:
            self._data.y["train"] = one_hot(self._data.y["train"], 10).astype("float32")
            self._data.y["valid"] = one_hot(self._data.y["valid"], 10).astype("float32")
            self._data.y["test"] = one_hot(self._data.y["test"], 10).astype("float32")

        self._data.x["train"] = self._data.x["train"].reshape(self._data.x["train"].shape[0],
                                                              self._data.x["train"].shape[1] *
                                                              self._data.x["train"].shape[2])
        self._data.x["valid"] = self._data.x["valid"].reshape(self._data.x["valid"].shape[0],
                                                              self._data.x["valid"].shape[1] *
                                                              self._data.x["valid"].shape[2])
        self._data.x["test"] = self._data.x["test"].reshape(self._data.x["test"].shape[0],
                                                              self._data.x["test"].shape[1] *
                                                              self._data.x["test"].shape[2])

    def reset_iterator(self):
        """Resets the data iterator and shuffles the examples."""

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

        Returns:
            output: a dictionary containing input 'x' and output 'y'.
        """

        if self._state.batches_done[tag] == self._state.batches[tag]:
            return None

        start_ind = self._state.batches_done[tag] * self._state.batch_size
        end_ind = (self._state.batches_done[tag] + 1) * self._state.batch_size
        if end_ind > len(self._data.x[tag]):
            end_ind = len(self._data.x[tag])

        indices = self._state.iter_list[tag][start_ind:end_ind]

        x = self._data.x[tag][indices]
        y = self._data.y[tag][indices]

        output = {}
        output['x'] = x
        output['y'] = y

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

    gen = MNISTData()

    data = None
    count = 0
    while True:
        data = gen.next("train")
        if data is not None:
            count += 1
            print(count)
        else:
            break
    print(count)

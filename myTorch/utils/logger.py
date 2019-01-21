import tensorflow as tf
from os import listdir
from os.path import isfile, join
from shutil import copyfile, rmtree

from myTorch.utils import create_folder


class Logger(object):
    """A generic class for logging in tensorboard."""

    def __init__(self, log_dir, backup=False):
        """Creates a summary writer logging to log_dir.

        Args:
            log_dir: str, absolute path to the directory to log.
            backup: bool, if True, tensorboard log is also backed up while checkpointing. Costly operation.
        """

        self._writer = tf.summary.FileWriter(log_dir)
        self._log_dir = log_dir
        self._backup = backup

    def log_scalar(self, tag, value, step):
        """Log a scalar variable.

        Args:
            tag: str, name of the metric to be logged.
            value: float, value to be logged.
            step: int, step value.
        """

        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self._writer.add_summary(summary, step)

    def save(self, dir_name):
        """Saves the current state of the log files.

        Args:
            dir_name: name of the directory to save the log files.
        """

        if self._backup:
            create_folder(dir_name)
            rmtree(dir_name)
            create_folder(dir_name)

            tf_path = self._log_dir
            onlyfiles = [f for f in listdir(tf_path) if isfile(join(tf_path, f))]

            for file in onlyfiles:
                copyfile(join(tf_path, file), join(dir_name, file))

    def load(self, dir_name):
        """Loads the log files from given directory.

        Args:
            dir_name: name of the directory to load the log file from.
        """

        if self._backup:
            rmtree(self._log_dir)
            create_folder(self._log_dir)

            onlyfiles = [f for f in listdir(dir_name) if isfile(join(dir_name, f))]

            for file in onlyfiles:
                copyfile(join(dir_name, file), join(self._log_dir, file))

    def force_restart(self):
        """Clears the log for force restart."""

        create_folder(self._log_dir)
        rmtree(self._log_dir)
        self._writer = tf.summary.FileWriter(self._log_dir)

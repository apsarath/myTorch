"""Implementation of a simple experiment class."""
import logging
import os
from shutil import rmtree, copyfile

from myTorch.utils import create_folder


class Experiment(object):
    """Implementation of a simple experiment class."""

    def __init__(self, name, dir_name):
        """Initializes an experiment object.

        Args:
            name: str, name of the experiment.
            dir_name: str, absolute path to the directory to save/load the experiment.
        """

        self._name = name
        self._dir_name = dir_name
        create_folder(self._dir_name)

        self._model = None
        self._config = None
        self._logger = None
        self._train_statistics = None
        self._data_iterator = None
        self._agent = None
        self._environment = None

    def register_experiment(self, model=None, config=None, logger=None, train_statistics=None, data_iterator=None,
                            agent=None, environment=None):
        """Registers all the components of an experiment.

        Args:
            model: a model object.
            config: a config gin file.
            logger: a logger object.
            train_statistics: a train_statistics dictionary.
            data_iterator: a data iterator object.
            agent: an agent object.
            environment: an environment object.
        """

        self._model = model
        self._config = config
        self._logger = logger
        self._train_statistics = train_statistics
        self._data_iterator = data_iterator
        self._agent = agent
        self._environment = environment

    def save(self, tag="current"):
        """Saves the experiment.
        Args:
            tag: str, tag to prefix the folder.
        """

        save_dir = os.path.join(self._dir_name, tag)
        create_folder(save_dir)

        logging.info("Saving the experiment at {}".format(save_dir))

        flag_file = os.path.join(save_dir, "flag.p")
        if os.path.isfile(flag_file):
            os.remove(flag_file)

        if self._model is not None:
            self._model.save(save_dir)

        if self._config is not None:
            file_name = os.path.join(save_dir, "config.p")
            self._config.save(file_name)

        if self._logger is not None:
            file_name = os.path.join(save_dir, "logger")
            self._logger.save(file_name)

        if self._train_statistics is not None:
            file_name = os.path.join(save_dir, "train_statistics.p")
            self._train_statistics.save(file_name)

        if self._data_iterator is not None:
            file_name = os.path.join(save_dir, "data_iterator.p")
            self._data_iterator.save(file_name)

        if self._agent is not None:
            self._agent.save(save_dir)

        if self._environment is not None:
            file_name = os.path.join(save_dir, "environment.p")
            self._environment.save(file_name)

        file = open(flag_file, "w")
        file.close()

    def is_resumable(self, tag="current"):
        """ Returns true if the experiment is resumable.

        Args:
            tag: str, tag for the saved experiment.
        """

        flag_file = os.path.join(self._dir_name, tag, "flag.p")
        if os.path.isfile(flag_file):
            return True
        else:
            return False

    def resume(self, tag="current"):
        """Resumes the experiment from a checkpoint.

        Args:
            tag: str, tag for the saved experiment.
        """

        if not self.is_resumable(tag):
            logging.warning("This experiment is not resumable!")
            logging.warning("Force restarting the experiment!")
            self.force_restart(tag)

        else:
            save_dir = os.path.join(self._dir_name, tag)
            logging.info("Loading the experiment from {}".format(save_dir))

            if self._model is not None:
                self._model.load(save_dir)

            if self._config is not None:
                file_name = os.path.join(save_dir, "config.p")
                self._config.load(file_name)

            if self._logger is not None:
                file_name = os.path.join(save_dir, "logger")
                self._logger.load(file_name)

            if self._train_statistics is not None:
                file_name = os.path.join(save_dir, "train_statistics.p")
                self._train_statistics.load(file_name)

            if self._data_iterator is not None:
                file_name = os.path.join(save_dir, "data_iterator.p")
                self._data_iterator.load(file_name)

            if self._agent is not None:
                self._agent.load(save_dir)

            if self._environment is not None:
                file_name = os.path.join(save_dir, "environment.p")
                self._environment.load(file_name)

    def force_restart(self):
        """Force restarting an experiment from beginning."""

        logging.info("Force restarting the experiment...")

        save_dir = os.path.join(self._dir_name)
        create_folder(save_dir)
        rmtree(save_dir)

        if self._logger is not None:
            self._logger.force_restart()

from copy import deepcopy

from myTorch import Experiment


class CurriculumExperiment(Experiment):
    """This imeplementation extends the base experiment class by supporting curriculum."""

    def __init__(self, name, dir_name):
        """Initializes an experiment object.

        Args:
            name: str, name of the experiment.
            dir_name: str, absolute path to the directory to save/load the experiment.
        """

        super().__init__(name=name, dir_name=dir_name)

    def get_curriculum_config(self):
        """Method to generate curriculum based on the registered config.
        For now, we generate the curriculum only using the `min_len` and `max_len` attributes
        I will look into how to extend it for other
        usecases"""
        min_seq_len = self._config.min_seq_len
        max_seq_len = self._config.max_seq_len
        step_seq_len = self._config.step_seq_len
        for seq_len in range(min_seq_len, max_seq_len+1, step_seq_len):
            curriculum_config = deepcopy(self._config)
            curriculum_config.seq_len = seq_len
            yield curriculum_config

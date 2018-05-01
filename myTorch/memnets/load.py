import os

from myTorch.utils import MyContainer
from myTorch.memnets.train import create_experiment


def load_experiment(save_dir):

    config = MyContainer()
    file_name = os.path.join(save_dir, "config.p")
    config.load(file_name)

    if config.device != "cpu":
        config.device = "cpu"

    if config.use_tflogger:
        config.use_tflogger = False

    if config.batch_size != 1:
        config.batch_size = 1

    experiment, model, data_iterator, tr, logger, device = create_experiment(config)

    experiment.resume()

    return experiment, model, data_iterator, device, config


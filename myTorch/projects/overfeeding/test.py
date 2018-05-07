import argparse
import logging

import numpy
import torch

from myTorch import Experiment
from myTorch.projects.overfeeding.recurrent_net import Recurrent
from myTorch.task.associative_recall_task import AssociativeRecallData
from myTorch.task.copy_task import CopyData
from myTorch.task.copying_memory import CopyingMemoryData
from myTorch.task.repeat_copy_task import RepeatCopyData
from myTorch.utils import create_config

parser = argparse.ArgumentParser(description="Algorithm Learning Task")
parser.add_argument("--config", type=str, default="config/default.yaml", help="config file path.")
parser.add_argument("--force_restart", type=bool, default=False, help="if True start training from scratch.")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)


def get_data_iterator(config):
    if config.task == "copy":
        data_iterator = CopyData(num_bits=config.num_bits, min_len=config.seq_len,
                                 max_len=config.seq_len, batch_size=config.batch_size)
    elif config.task == "repeat_copy":
        data_iterator = RepeatCopyData(num_bits=config.num_bits, min_len=config.seq_len,
                                       max_len=config.seq_len, min_repeat=config.min_repeat,
                                       max_repeat=config.max_repeat, batch_size=config.batch_size)
    elif config.task == "associative_recall":
        data_iterator = AssociativeRecallData(num_bits=config.num_bits, min_len=config.seq_len,
                                              max_len=config.seq_len, block_len=config.block_len,
                                              batch_size=config.batch_size)
    elif config.task == "copying_memory":
        data_iterator = CopyingMemoryData(seq_len=config.seq_len, time_lag=config.time_lag,
                                          batch_size=config.batch_size, seed=config.seed)
    return data_iterator



def test_make_net_wider_op(model, data_iterator, device, batch_size=32, error_threshold=1e-5):
    """Method to test the `make_net_wider` operation.

    Args:
        model: model object
        data_iterator: data iterator object
        device: torch device
        batch_size: batch size
        error_threshold: acceptable error threshold
    """

    data = data_iterator.next()
    test_passed = True

    for i in range(0, data["datalen"]):

        h = model.reset_hidden_randomly(batch_size=batch_size)
        x = torch.from_numpy(numpy.asarray(data['x'][i])).to(device)
        output_original = model(x)
        model.set_hidden(h)
        model.make_net_wider(new_hidden_dim=512)
        model = model.to(device)
        output_after_widen = model(x)
        error = torch.sum(torch.abs(output_after_widen - output_original))
        if (error > error_threshold):
            logging.info("make_net_wider operation failed. Error = {}, acceptable threshold error = {}"
                         .format(error, error_threshold))
            test_passed = False
        return test_passed


def run_test_experiment():
    """Runs a test experiment to validate that the `make_net_wider` operation works"""

    config = create_config(args.config)

    logging.info(config.get())

    device = torch.device(config.device)
    logging.info("using {}".format(config.device))

    experiment = Experiment(config.name, config.save_dir)
    experiment.register_config(config)

    torch.manual_seed(config.rseed)

    model = Recurrent(device, config.input_size, config.output_size,
                      num_layers=config.num_layers, layer_size=config.layer_size,
                      cell_name=config.model, activation=config.activation,
                      output_activation="linear").to(device)
    experiment.register_model(model)

    data_iterator = get_data_iterator(config)
    experiment.register_data_iterator(data_iterator)

    max_steps = 1000
    test_passed = True
    for step in range(max_steps):
        test_passed = test_make_net_wider_op(model, data_iterator, device, batch_size=config.batch_size,
                                             error_threshold=1e-5)
        if not test_passed:
            break
    if (test_passed):
        logging.info("All tests passed")


if __name__ == '__main__':
    run_test_experiment()

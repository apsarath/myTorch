import numpy
import argparse
import logging

import torch

from myTorch import Experiment
from myTorch.memnets.recurrent_net import Recurrent
from myTorch.task.copy_task import CopyData
from myTorch.task.repeat_copy_task import RepeatCopyData
from myTorch.task.associative_recall_task import AssociativeRecallData
from myTorch.utils.logging import Logger
from myTorch.utils import MyContainer, get_optimizer, create_config
import torch.nn.functional as F

parser = argparse.ArgumentParser(description="Algorithm Learning Task")
parser.add_argument("--config", type=str, default="config/default.yaml", help="config file path.")
parser.add_argument("--force_restart", type=bool, default=False, help="if True start training from scratch.")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)


def get_data_iterator(config):

    if config.task == "copy":
        data_iterator = CopyData(num_bits=config.num_bits, min_len=config.min_len,
                                 max_len=config.max_len, batch_size=config.batch_size)
    elif config.task == "repeat_copy":
        data_iterator = RepeatCopyData(num_bits=config.num_bits, min_len=config.min_len,
                                       max_len=config.max_len, min_repeat=config.min_repeat,
                                       max_repeat=config.max_repeat, batch_size=config.batch_size)
    elif config.task == "associative_recall":
        data_iterator = AssociativeRecallData(num_bits=config.num_bits, min_len=config.min_len,
                                              max_len=config.max_len, block_len=config.block_len,
                                              batch_size=config.batch_size)
    return data_iterator


def train(experiment, model, config, data_iterator, tr, logger, device):
    """Training loop.

    Args:
        experiment: experiment object.
        model: model object.
        config: config dictionary.
        data_iterator: data iterator object
        tr: training statistics dictionary.
        logger: logger object.
    """

    for step in range(tr.updates_done, config.max_steps):

        data = data_iterator.next()
        seqloss = 0

        model.reset_hidden(batch_size=config.batch_size)

        for i in range(0, data["datalen"]):

            x = torch.from_numpy(numpy.asarray(data['x'][i])).to(device)
            y = torch.from_numpy(numpy.asarray(data['y'][i])).to(device)
            mask = float(data["mask"][i])

            model.optimizer.zero_grad()

            output = model(x)
            loss = F.binary_cross_entropy_with_logits(output, y)
            seqloss += (loss * mask)

        seqloss /= sum(data["mask"])
        tr.average_bce.append(seqloss.item())
        running_average = sum(tr.average_bce) / len(tr.average_bce)

        if config.use_tflogger:
            logger.log_scalar("loss", running_average, step + 1)

        seqloss.backward(retain_graph=False)

        for param in model.parameters():
            param.grad.clamp_(config.grad_clip[0], config.grad_clip[1])

        model.optimizer.step()

        tr.updates_done +=1
        if tr.updates_done % 100 == 0:
            logging.info("examples seen: {}, running average of BCE: {}".format(tr.updates_done*config.batch_size,
                                                                                running_average))
        if tr.updates_done % config.save_every_n == 0:
            experiment.save()


def run_experiment():
    """Runs the experiment."""

    config = create_config(args.config)

    logging.info(config.get())

    device = torch.device(config.device)
    logging.info("using {}".format(config.device))

    experiment = Experiment(config.name, config.save_dir)
    experiment.register_config(config)

    logger = None
    if config.use_tflogger:
        logger = Logger(config.tflog_dir)
        experiment.register_logger(logger)

    torch.manual_seed(config.rseed)

    model = Recurrent(device, config.input_size, config.output_size,
                      num_layers=config.num_layers, layer_size=config.layer_size,
                      cell_name=config.model, activation=config.activation,
                      output_activation="linear").to(device)
    experiment.register_model(model)

    data_iterator = get_data_iterator(config)
    experiment.register_data_iterator(data_iterator)

    optimizer = get_optimizer(model.parameters(), config)
    model.register_optimizer(optimizer)

    tr = MyContainer()
    tr.updates_done = 0
    tr.average_bce = []
    experiment.register_train_statistics(tr)

    if not args.force_restart:
        if experiment.is_resumable():
            experiment.resume()
    else:
        experiment.force_restart()

    train(experiment, model, config, data_iterator, tr, logger, device)


if __name__ == '__main__':
    run_experiment()

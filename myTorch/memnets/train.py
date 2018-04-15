import numpy
import argparse
import logging

import torch
from torch.autograd import Variable

from myTorch import Experiment
from myTorch.memnets.recurrent_net import Recurrent
from myTorch.task.copy_task import CopyData
from myTorch.task.repeat_copy_task import RepeatCopyData
from myTorch.utils.logging import Logger
from myTorch.utils import MyContainer, get_optimizer
import torch.nn.functional as F
from myTorch.memnets.config import *

parser = argparse.ArgumentParser(description="Algorithm Learning Task")
parser.add_argument("--config", type=str, default="copy_task_RNN", help="config name")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)


def train(experiment, model, config, data_iterator, tr, logger):
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

            x = Variable(torch.from_numpy(numpy.asarray(data['x'][i])))
            y = Variable(torch.from_numpy(numpy.asarray(data['y'][i])))
            if config.use_gpu:
                x = x.cuda()
                y = y.cuda()
            mask = float(data["mask"][i])

            model.optimizer.zero_grad()

            output = model(x)
            loss = F.binary_cross_entropy_with_logits(output, y)
            seqloss += (loss * mask)

        seqloss /= sum(data["mask"])
        tr.average_bce.append(seqloss.data[0])
        running_average = sum(tr.average_bce) / len(tr.average_bce)

        if config.use_tflogger:
            logger.log_scalar("loss", running_average, step + 1)

        seqloss.backward(retain_graph=False)

        for param in model.parameters():
            param.grad.data.clamp_(config.grad_clip[0], config.grad_clip[1])

        model.optimizer.step()

        tr.updates_done +=1
        if tr.updates_done % 1000 == 0:
            logging.info("examples seen: {}, running average of BCE: {}".format(tr.updates_done*config.batch_size,
                                                                                running_average))
        if tr.updates_done % config.save_every_n == 0:
            experiment.save()


def run_experiment():
    """Runs the experiment."""

    config = eval(args.config)()

    logging.info(config.get())

    experiment = Experiment(config.name, config.save_dir)
    experiment.register_config(config)

    logger = None
    if config.use_tflogger:
        logger = Logger(config.tflog_dir)
        experiment.register_logger(logger)

    torch.manual_seed(config.rseed)

    model = Recurrent(config.input_size, config.output_size,
                      num_layers=config.num_layers, layer_size=config.layer_size,
                      cell_name=config.model, activation=config.activation,
                      output_activation="linear", use_gpu=config.use_gpu)
    experiment.register_model(model)
    if config.use_gpu:
        model.cuda()

    if config.task == "copy":
        data_iterator = CopyData(num_bits=config.num_bits, min_len=config.min_len,
                                 max_len=config.max_len, batch_size=config.batch_size)
    elif config.task == "repeat_copy":
        data_iterator = RepeatCopyData(num_bits=config.num_bits, min_len=config.min_len,
                                       max_len=config.max_len, min_repeat=config.min_repeat,
                                       max_repeat=config.max_repeat, batch_size=config.batch_size)

    experiment.register_data_iterator(data_iterator)

    optimizer = get_optimizer(model.parameters(), config)
    model.register_optimizer(optimizer)

    tr = MyContainer()
    tr.updates_done = 0
    tr.average_bce = []
    experiment.register_train_statistics(tr)

    if not config.force_restart:
        if experiment.is_resumable():
            experiment.resume()
    else:
        experiment.force_restart()

    train(experiment, model, config, data_iterator, tr, logger)


if __name__ == '__main__':
    run_experiment()

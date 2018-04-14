import numpy
import argparse
import logging

import torch
from torch.autograd import Variable

from myTorch import Experiment
from myTorch.memnets.recurrent_net import Recurrent
from myTorch.task.copy_task import *
from myTorch.utils.logging import Logger
from myTorch.utils import MyContainer, get_optimizer
import torch.nn.functional as F


from myTorch.memnets.config import *

parser = argparse.ArgumentParser(description="Algorithm Learning Task")
parser.add_argument("--config", type=str, default="copy_task_RNN", help="config name")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)


def train(experiment, model, config, data_iterator, tr, logger):

    for step in range(tr.examples_seen, config.max_steps):

        data = data_iterator.next()
        seqloss = 0

        for i in range(0, data["datalen"]):

            x = Variable(torch.from_numpy(numpy.asarray([data['x'][i]])))
            y = Variable(torch.from_numpy(numpy.asarray([data['y'][i]])))
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
        logging.info("running average of BCE: {}".format(running_average))
        if config.use_tflogger:
            logger.log_scalar("loss", running_average, step + 1)

        seqloss.backward(retain_graph=False)

        for param in model.parameters():
            param.grad.data.clamp_(config.grad_clip[0], config.grad_clip[1])

        model.optimizer.step()

        model.reset_hidden()
        tr.examples_seen += 1

        if tr.examples_seen % 100 == 0:
            experiment.save()


def run_experiment():
    """Runs the experiment."""

    config = eval(args.config)()

    print(config.get())

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

    data_iterator = CopyDataGen(num_bits=config.num_bits, min_len=config.min_len,
                                max_len=config.max_len)
    experiment.register_data_iterator(data_iterator)

    optimizer = get_optimizer(model.parameters(), config)
    model.register_optimizer(optimizer)

    tr = MyContainer()
    tr.examples_seen = 0
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

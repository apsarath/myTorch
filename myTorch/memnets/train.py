import numpy
import argparse
import logging

import torch

from myTorch import Experiment
from myTorch.memnets.recurrent_net import Recurrent
from myTorch.task.copy_task import CopyData
from myTorch.task.repeat_copy_task import RepeatCopyData
from myTorch.task.associative_recall_task import AssociativeRecallData
from myTorch.task.copying_memory import CopyingMemoryData
from myTorch.task.adding_task import AddingData
from myTorch.task.denoising import DenoisingData
from myTorch.utils.logging import Logger
from myTorch.utils import MyContainer, get_optimizer, create_config
import torch.nn.functional as F

parser = argparse.ArgumentParser(description="Algorithm Learning Task")
parser.add_argument("--config", type=str, default="config/default.yaml", help="config file path.")
parser.add_argument("--force_restart", type=bool, default=False, help="if True start training from scratch.")


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
    elif config.task == "copying_memory":
        data_iterator = CopyingMemoryData(seq_len=config.seq_len, time_lag_min=config.time_lag_min,
                                          time_lag_max=config.time_lag_max, num_digits=config.num_digits,
                                          num_noise_digits=config.num_noise_digits, 
                                          batch_size=config.batch_size, seed=config.seed)
    elif config.task == "adding":
        data_iterator = AddingData(seq_len=config.seq_len, batch_size=config.batch_size, seed=config.seed)
    elif config.task == "denoising_copy":
        data_iterator = DenoisingData(seq_len=config.seq_len, time_lag_min=config.time_lag_min, 
                                      time_lag_max=config.time_lag_max, batch_size=config.batch_size, 
                                      num_noise_digits=config.num_noise_digits, 
                                      num_digits=config.num_digits, seed=config.seed)

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

        if config.inter_saving is not None:
            if tr.updates_done in config.inter_saving:
                experiment.save(str(tr.updates_done))

        data = data_iterator.next()
        seqloss = 0

        model.reset_hidden(batch_size=config.batch_size)

        for i in range(0, data["datalen"]):

            x = torch.from_numpy(numpy.asarray(data['x'][i])).to(device)
            y = torch.from_numpy(numpy.asarray(data['y'][i])).to(device)
            mask = float(data["mask"][i])

            model.optimizer.zero_grad()

            output = model(x)
            if config.task == "copying_memory" or config.task == "denoising_copy":
                loss = F.cross_entropy(output, y.squeeze(1))
            elif config.task == "adding":
                loss = F.mse_loss(output, y)
            else:
                loss = F.binary_cross_entropy_with_logits(output, y)

            seqloss += (loss * mask)

        seqloss /= sum(data["mask"])
        tr.average_bce.append(seqloss.item())
        running_average = sum(tr.average_bce) / len(tr.average_bce)

        if config.use_tflogger:
            logger.log_scalar("running_avg_loss", running_average, step + 1)
            logger.log_scalar("loss", tr.average_bce[-1], step + 1)

        seqloss.backward(retain_graph=False)

        total_norm = torch.nn.utils.clip_grad_norm(model.parameters(), config.grad_clip_norm)

        if config.use_tflogger:
            logger.log_scalar("inst_total_norm", total_norm, step + 1)
 

        model.optimizer.step()

        tr.updates_done +=1
        if tr.updates_done % 1 == 0:
            logging.info("examples seen: {}, inst loss: {}, total_norm : {}".format(tr.updates_done*config.batch_size,
                                                                                tr.average_bce[-1], total_norm))
        if tr.updates_done % config.save_every_n == 0:
            experiment.save()


def create_experiment(config):
    """Creates an experiment based on config."""

    device = torch.device(config.device)
    logging.info("using {}".format(config.device))

    experiment = Experiment(config.name, config.save_dir)
    experiment.register_config(config)

    logger = None
    if config.use_tflogger:
        logger = Logger(config.tflog_dir)
        experiment.register_logger(logger)

    torch.manual_seed(config.rseed)
    input_size = config.num_digits + config.num_noise_digits + 1
    output_size = input_size - 1
    t_max = 1 + config.time_lag_max + config.seq_len

    model = Recurrent(device, input_size, output_size,
                      num_layers=config.num_layers, layer_size=config.layer_size,
                      cell_name=config.model, activation=config.activation,
                      output_activation="linear", layer_norm=config.layer_norm,
                      identity_init=config.identity_init, chrono_init=config.chrono_init,
                      t_max=t_max, use_relu=config.use_relu).to(device)
    experiment.register_model(model)

    data_iterator = get_data_iterator(config)
    experiment.register_data_iterator(data_iterator)

    optimizer = get_optimizer(model.parameters(), config)
    model.register_optimizer(optimizer)

    tr = MyContainer()
    tr.updates_done = 0
    tr.average_bce = []
    experiment.register_train_statistics(tr)

    return experiment, model, data_iterator, tr, logger, device


def run_experiment(args):
    """Runs the experiment."""

    config = create_config(args.config)

    logging.info(config.get())

    experiment, model, data_iterator, tr, logger, device = create_experiment(config)

    if not args.force_restart:
        if experiment.is_resumable():
            experiment.resume()
    else:
        experiment.force_restart()

    train(experiment, model, config, data_iterator, tr, logger, device)


if __name__ == '__main__':
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    run_experiment(args)

import argparse
import logging

import numpy
import numpy as np
import torch
import torch.nn.functional as F

from myTorch import Experiment
from myTorch.projects.overfeeding.recurrent_net import Recurrent
from myTorch.projects.overfeeding.utils.curriculum import curriculum_generator
from myTorch.projects.overfeeding.utils.metric import get_metric_registry
from myTorch.task.associative_recall_task import AssociativeRecallData
from myTorch.task.copy_task import CopyData
from myTorch.task.copying_memory import CopyingMemoryData
from myTorch.task.repeat_copy_task import RepeatCopyData
from myTorch.utils import MyContainer, get_optimizer, create_config
from myTorch.utils.logging import Logger

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


def train(experiment, model, config, data_iterator, tr, logger, device, metrics):
    """Training loop.

    Args:
        experiment: experiment object.
        model: model object.
        config: config dictionary.
        data_iterator: data iterator object
        tr: training statistics dictionary.
        logger: logger object.
    """
    should_stop_curriculum = True
    for step in range(tr.updates_done, config.max_steps):

        data = data_iterator.next()
        seqloss = 0
        average_accuracy = 0

        model.reset_hidden(batch_size=config.batch_size)

        for i in range(0, data["datalen"]):

            x = torch.from_numpy(numpy.asarray(data['x'][i])).to(device)
            y = torch.from_numpy(numpy.asarray(data['y'][i])).to(device)
            mask = float(data["mask"][i])

            model.optimizer.zero_grad()

            output = model(x)
            if config.task == "copying_memory":
                loss = F.torch.nn.functional.cross_entropy(output, y.squeeze(1))
            else:
                loss = F.binary_cross_entropy_with_logits(output, y)

            seqloss += (loss * mask)
            predictions = F.softmax(
                (torch.cat(
                    ((1 - output).unsqueeze(2), output.unsqueeze(2)),
                    dim=2))
                , dim=2)
            predictions = predictions.max(2)[1].float()
            average_accuracy += ((y == predictions).int().sum().item() * mask)

        seqloss /= sum(data["mask"])
        average_accuracy /= sum(data["mask"])
        x_shape = data["x"].shape
        average_accuracy /= (x_shape[1] * x_shape[2])
        tr.average_bce.append(seqloss.item())
        tr.average_accuracy.append(average_accuracy)
        running_average_bce = sum(tr.average_bce) / len(tr.average_bce)
        running_average_accuracy = sum(tr.average_accuracy) / len(tr.average_accuracy)

        if config.use_tflogger:
            logger.log_scalar("running_avg_loss", running_average_bce, step + 1)
            logger.log_scalar("loss", tr.average_bce[-1], step + 1)
            logger.log_scalar("average accuracy", average_accuracy, step + 1)
            logger.log_scalar("running_average_accuracy", running_average_accuracy, step + 1)

        seqloss.backward(retain_graph=False)

        for param in model.parameters():
            param.grad.clamp_(config.grad_clip[0], config.grad_clip[1])

        model.optimizer.step()

        metrics["loss"].update(tr.average_bce[-1])
        metrics["accuracy"].update(tr.average_accuracy[-1])

        tr.updates_done += 1
        if tr.updates_done % 1 == 0:
            logging.info("examples seen: {}, running average of BCE: {}, "
                         "average accuracy for last batch: {}, "
                         "running average of accuracy: {}".format(tr.updates_done * config.batch_size,
                                                                  running_average_bce,
                                                                  average_accuracy,
                                                                  running_average_accuracy))
        if tr.updates_done % config.save_every_n == 0:
            experiment.save()

        if (metrics["accuracy"].is_best_so_far()):
            experiment.save(tag="best")

        if (metrics["loss"].should_stop_early() or metrics["accuracy"].should_stop_early()):
            logging.info("Early stopping after {} epochs".format(step))
            logging.info("Loss = {} for the best performing model".format(metrics["loss"].get_best_so_far()))
            logging.info("Accuracy = {} for the best performing model".format(metrics["accuracy"].get_best_so_far()))
            average_accuracy_array = np.asarray(tr.average_accuracy)[-config.average_over_last_n:]
            if (np.mean(average_accuracy_array) > 0.8):
                should_stop_curriculum = False
            break

    return should_stop_curriculum


def train_curriculum():
    """Runs the experiment."""

    config = create_config(args.config)

    logging.info(config.get())

    device = torch.device(config.device)
    logging.info("using device {}".format(config.device))

    torch.manual_seed(config.rseed)

    model = Recurrent(device, config.input_size, config.output_size,
                      num_layers=config.num_layers, layer_size=config.layer_size,
                      cell_name=config.model, activation=config.activation,
                      output_activation="linear").to(device)

    optimizer = get_optimizer(model.parameters(), config)
    model.register_optimizer(optimizer)

    should_stop_curriculum = False
    experiment = Experiment(config.name, config.save_dir)
    logger = None
    if config.use_tflogger:
        logger = Logger(config.tflog_dir)
        experiment.register_logger(logger)
    experiment.register_model(model)
    experiment.register_config(config)

    # This part might cause some issues later
    if not args.force_restart:
        if experiment.is_resumable():
            experiment.resume()
            logging.info("Resuming experiment")
        else:
            logging.info("Restarting the experiment")
    else:
        experiment.force_restart()
        logging.info("Forced to restart the experiment")

    for curriculum_config in curriculum_generator(config):
        logging.info("Starting curriculum with seq_len: {}".format(curriculum_config.seq_len))
        data_iterator = get_data_iterator(curriculum_config)
        experiment.register_data_iterator(data_iterator)
        tr = MyContainer()
        tr.updates_done = 0
        tr.average_bce = []
        tr.average_accuracy = []
        experiment.register_train_statistics(tr)
        metrics = get_metric_registry(time_span=curriculum_config.time_span)
        should_stop_curriculum = train(experiment, model, curriculum_config, data_iterator, tr, logger, device, metrics)
        if (should_stop_curriculum):
            logging.info("Stopping curriculum after seq_len: {}".format(curriculum_config.seq_len))
            break


if __name__ == '__main__':
    train_curriculum()

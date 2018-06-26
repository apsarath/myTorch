import logging

import torch

from myTorch import Experiment
from myTorch.projects.overfeeding.recurrent_net import Recurrent
from myTorch.task.associative_recall_task import AssociativeRecallData
from myTorch.task.copy_task import CopyData
from myTorch.task.copying_memory import CopyingMemoryData
from myTorch.task.repeat_copy_task import RepeatCopyData
from myTorch.utils import get_optimizer
from myTorch.utils.logging import Logger
from myTorch.projects.overfeeding.gem import GemModel


def get_data_iterator(config, seed=None):
    if not seed:
        seed = config.rseed
    if config.task == "copy":
        data_iterator = CopyData(num_bits=config.num_bits, min_len=config.seq_len,
                                 max_len=config.seq_len, batch_size=config.batch_size, seed=seed)
    elif config.task == "repeat_copy":
        data_iterator = RepeatCopyData(num_bits=config.num_bits, min_len=config.seq_len,
                                       max_len=config.seq_len, min_repeat=config.min_repeat,
                                       max_repeat=config.max_repeat, batch_size=config.batch_size, seed=seed)
    elif config.task == "associative_recall":
        data_iterator = AssociativeRecallData(num_bits=config.num_bits, min_len=config.seq_len,
                                              max_len=config.seq_len, block_len=config.block_len,
                                              batch_size=config.batch_size, seed=seed)
    elif config.task == "copying_memory":
        data_iterator = CopyingMemoryData(seq_len=config.seq_len, time_lag=config.time_lag,
                                          batch_size=config.batch_size, seed=config.seed)
    return data_iterator


def prepare_experiment(config):
    device = torch.device(config.device)
    logging.info("using device {}".format(config.device))

    torch.manual_seed(config.rseed)


    if config.use_gem:
        model = GemModel(
        device,
        config.input_size,
        config.output_size,
        num_layers=config.num_layers,
        layer_size=config.layer_size,
        cell_name=config.model,
        activation=config.activation,
        output_activation="linear",
        n_tasks = int((config.max_seq_len - config.min_seq_len)/config.step_seq_len) + 1,
        memory_strength = config.memory_strength,
        num_memories = config.num_memories,
        task=config.task
    )
    else:
        model = Recurrent(device, config.input_size, config.output_size,
                          num_layers=config.num_layers, layer_size=config.layer_size,
                          cell_name=config.model, activation=config.activation,
                          output_activation="linear", task=config.task)

    model = model.to(device)
    optimizer = get_optimizer(model.parameters(), config)
    model.register_optimizer(optimizer)
    experiment = Experiment(config.name, config.save_dir)
    if config.use_tflogger:
        logger = Logger(config.tflog_dir)
        experiment.register_logger(logger)
    experiment.register_model(model)
    experiment.register_config(config)
    experiment.register_device(device)
    return experiment

def try_to_restart_experiment(experiment, force_restart):
    if not force_restart:
        if experiment.is_resumable():
            experiment.resume()
            logging.info("Resuming experiment")
        else:
            logging.info("Restarting the experiment")
    else:
        experiment.force_restart()
        logging.info("Forced to restart the experiment")
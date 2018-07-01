import logging

import torch

from myTorch import Experiment
from myTorch.projects.overfeeding.gem import GemModel
from myTorch.projects.overfeeding.recurrent_net import Recurrent
from myTorch.task.associative_recall_task import AssociativeRecallData
from myTorch.task.copy_task import CopyData
from myTorch.task.copying_memory import CopyingMemoryData
from myTorch.task.repeat_copy_task import RepeatCopyData
from myTorch.utils import get_optimizer
from myTorch.utils.logging import Logger
from copy import deepcopy


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

    model = choose_model(config, device)

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


def expand_model(experiment, config, step):
    model = experiment._model
    device = experiment._device
    logger = experiment._logger
    is_expansion_successful = False
    old_config = experiment._config

    if(model.can_make_net_wider(expanded_layer_size=config.expanded_layer_size, expansion_offset=config.expansion_offset)):
        # Lets expand
        previous_layer_size = model.layer_size
        experiment.save(tag="model_before_expanding")
        previous_optimizer_state_dict = deepcopy(model.optimizer.state_dict())
        model.make_net_wider(expanded_layer_size=config.expanded_layer_size,
                             expansion_offset=config.expansion_offset,
                             can_make_optimizer_wider=config.make_optimizer_wider,
                             use_noise=config.use_noise,
                             use_random_noise=config.use_random_noise)

        new_layer_size = model.layer_size
        new_config = deepcopy(config)
        new_config.lr = new_config.new_lr
        new_config.layer_size = new_layer_size

        wider_model = choose_model(new_config, device)

        if (config.expand_model_weights):
            # load the expanded weights
            wider_model.load_state_dict(model.state_dict())

        optimizer = get_optimizer(wider_model.parameters(), config)

        log_message_value = "Model index: {}. " \
                            "Curriculum index: {}. " \
                            "Previous learning rate {}. " \
                            "New learning rate {}," \
                            "step: {}".format(
            experiment.current_model_index,
            experiment.current_curriculum_index,
            config.lr,
            new_config.lr,
            step)

        log_message_tag = "New learning rate for the optimizer"

        if config.use_tflogger:
            logger.log_text(tag=log_message_tag,
                            value=log_message_value
                            )
        logging.info(log_message_tag + ": " + log_message_value)

        if (config.make_optimizer_wider):
            # get new optimizer and do the standard bookkeeping tasks
            prev_param_state_values = list(previous_optimizer_state_dict['state'].values())
            param_names_in_new_optimizer = optimizer.state_dict()['param_groups'][0]['params']
            for index, param in enumerate(optimizer.param_groups[0]['params']):
                new_value = prev_param_state_values[index]
                new_value['exp_avg'] = new_value['exp_avg'].to(device)
                new_value['exp_avg_sq'] = new_value['exp_avg_sq'].to(device)
                optimizer.state[0][param_names_in_new_optimizer[index]] = new_value
            log_message_value = "Model index: {}. " \
                                "Curriculum index: {}. " \
                                "Previous learning rate {}. " \
                                "New learning rate {}," \
                                "step: {}".format(
                experiment.current_model_index,
                experiment.current_curriculum_index,
                config.lr,
                new_config.lr,
                step)

            log_message_tag = "Widening Optimizer"

            if config.use_tflogger:
                logger.log_text(tag=log_message_tag,
                                value=log_message_value
                                )
            logging.info(log_message_tag + ": " + log_message_value)

        model = wider_model.to(device)
        model.register_optimizer(optimizer)
        experiment.register_model(model)

        experiment.save(tag="model_after_expanding")

        # Now we will reset the counters and continue training
        # metrics["loss"].reset()
        # metrics["accuracy"].reset()
        # metrics["loss"].make_timeless()
        # metrics["accuracy"].make_timeless()

        log_message_tag = "Widening model (expand_model_weights = {})" \
            .format(config.expand_model_weights)
        log_message_value = "Model index: {}. " \
                            "Curriculum index: {}. " \
                            "Previous learning rate {}. " \
                            "New learning rate {}," \
                            "step: {}".format(
            experiment.current_model_index,
            experiment.current_curriculum_index,
            config.lr,
            new_config.lr,
            step)

        if config.use_tflogger:
            logger.log_text(tag=log_message_tag,
                            value=log_message_value
                            )
        logging.info(log_message_tag + ": " + log_message_value)
        is_expansion_successful = True

    # else:
        # There is no scope to grow the model
        # metrics["loss"].make_timeless()
        # metrics["accuracy"].make_timeless()
        # Time to meet the creator
        # should_stop_curriculum = True
        # logging.info("When training model, model_index: {}, early stopping after {} epochs".format(
        #     experiment.current_model_index, step))
        # logging.info("When training model, model_index: {}, loss = {} for the best performing model".format(
        #     experiment.current_model_index, metrics["loss"].get_best_so_far()))
        # logging.info(
        #     "When training model, model_index: {}, accuracy = {} for the best performing model".format
        #     (experiment.current_model_index, metrics["accuracy"].get_best_so_far()))

    return  experiment, is_expansion_successful

def choose_model(config, device):
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
            n_tasks=int((config.max_seq_len - config.min_seq_len) / config.step_seq_len) + 1,
            memory_strength=config.memory_strength,
            num_memories=config.num_memories,
            task=config.task,
            use_regularisation=config.use_regularisation,
            regularisation_constant=config.regularisation_constant
        )
    else:
        model = Recurrent(device, config.input_size, config.output_size,
                          num_layers=config.num_layers, layer_size=config.layer_size,
                          cell_name=config.model, activation=config.activation,
                          output_activation="linear", task=config.task)

    return model
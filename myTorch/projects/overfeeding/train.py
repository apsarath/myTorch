import argparse
import logging
from copy import deepcopy

import numpy
import torch
import torch.nn.functional as F

from myTorch import Experiment
from myTorch.projects.overfeeding.recurrent_net import Recurrent
from myTorch.projects.overfeeding.utils.curriculum import curriculum_generator
from myTorch.projects.overfeeding.utils.metric import get_metric_registry
from myTorch.projects.overfeeding.utils.spectral import *
from myTorch.task.associative_recall_task import AssociativeRecallData
from myTorch.task.copy_task import CopyData
from myTorch.task.copying_memory import CopyingMemoryData
from myTorch.task.repeat_copy_task import RepeatCopyData
from myTorch.utils import MyContainer, get_optimizer, create_config, compute_grad_norm
from myTorch.utils.logging import Logger

parser = argparse.ArgumentParser(description="Algorithm Learning Task")
# parser.add_argument("--config", type=str, default="config/shagun/associative_recall.yaml", help="config file path.")
parser.add_argument("--config", type=str, default="config/default.yaml", help="config file path.")
parser.add_argument("--force_restart", type=bool, default=True, help="if True start training from scratch.")
args = parser.parse_args()


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


def train(experiment, model, config, data_iterator, tr, logger, device, metrics, model_idx):
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

        seqloss.backward(retain_graph=False)

        for param in model.parameters():
            param.grad.clamp_(config.grad_clip[0], config.grad_clip[1])

        gradient_norm = compute_grad_norm(parameters=model.parameters()).item()

        if tr.updates_done % 10 == 0:
            if config.use_tflogger:
                for key in ['_W_h2o',
                            '_list_of_modules.0._W_x2i',
                            '_list_of_modules.0._W_h2i',
                            '_list_of_modules.0._W_h2f',
                            '_list_of_modules.0._W_h2o',
                            '_list_of_modules.0._W_h2c']:
                    A = model.state_dict()[key].cpu().numpy()
                    spectral_metrics = compute_spectral_properties(A)
                    logger.log_scalar("model_index__{}__condition_number__{}".format(model_idx, key),
                                      spectral_metrics.condition_number, step + 1)
                    logger.log_scalar("model_index__{}__spectral_norm__{}".format(model_idx, key) + str(model_idx),
                                      spectral_metrics.spectral_norm, step + 1)
                    # logger.log_scalar("spectral_radius_{}_".format(key) + str(model_idx),
                    #                   spectral_metrics.spectral_radius, step + 1)

                logger.log_scalar("train_running_avg_loss_model_idx_" + str(model_idx), running_average_bce, step + 1)
                logger.log_scalar("train_loss_model_idx_" + str(model_idx), tr.average_bce[-1], step + 1)
                logger.log_scalar("train_average accuracy_model_idx_" + str(model_idx), average_accuracy, step + 1)
                logger.log_scalar("train_running_average_accuracy_model_idx_" + str(model_idx),
                                  running_average_accuracy,
                                  step + 1)
                logger.log_scalar("train_gradient_norm_model_idx_" + str(model_idx), gradient_norm,
                                  step + 1)

        model.optimizer.step()

        metrics["loss"].update(tr.average_bce[-1])
        metrics["accuracy"].update(tr.average_accuracy[-1])

        tr.updates_done += 1
        if tr.updates_done % 1 == 0:
            logging.info("When training model, model_index: {}, "
                         "examples seen: {}, "
                         "running average of BCE: {}, "
                         "average accuracy for last batch: {}, "
                         "running average of accuracy: {},"
                         "gradient norm: {}".format(model_idx, tr.updates_done * config.batch_size,
                                                    running_average_bce,
                                                    average_accuracy,
                                                    running_average_accuracy,
                                                    gradient_norm))
        if tr.updates_done % config.save_every_n == 0:
            experiment.save()

        if (metrics["accuracy"].is_best_so_far()):
            experiment.save(tag="best")

        if (tr.updates_done > config.average_over_last_n):
            average_accuracy_array = np.asarray(tr.average_accuracy)[-config.average_over_last_n:]
            if (np.mean(average_accuracy_array) > 0.8):
                # Could complete the task. No need to grow the model
                should_stop_curriculum = False
                break

            elif (metrics["loss"].should_stop_early() or metrics["accuracy"].should_stop_early()):
                # Could not complete the task
                # Lets try to grow:
                if (model.can_make_net_wider(expanded_layer_size=config.expanded_layer_size)):
                    # Lets expand
                    previous_layer_size = model.layer_size
                    experiment.save(tag="model_before_expanding")
                    model.make_net_wider(expanded_layer_size=config.expanded_layer_size,
                                         can_make_optimizer_wider=config.make_optimizer_wider)
                    previous_optimizer_state_dict = deepcopy(model.optimizer.state_dict())
                    new_layer_size = model.layer_size

                    wider_model = Recurrent(device, config.input_size, config.output_size,
                                            num_layers=config.num_layers, layer_size=new_layer_size,
                                            cell_name=config.model, activation=config.activation,
                                            output_activation="linear")

                    if (config.expand_model_weights):
                        # load the expanded weights
                        wider_model.load_state_dict(model.state_dict())

                    new_config = deepcopy(config)
                    new_config.lr = new_config.new_lr
                    optimizer = get_optimizer(wider_model.parameters(), config)

                    log_message_value = "Model index: {}. " \
                                        "Previous learning rate {}. " \
                                        "New learning rate {}," \
                                        "step: {}".format(
                        model_idx,
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
                                            "Previous size {}. " \
                                            "New size {}," \
                                            "step: {}".format(
                            model_idx,
                            previous_layer_size,
                            new_layer_size,
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
                    metrics["loss"].reset()
                    metrics["accuracy"].reset()
                    metrics["loss"].make_timeless()
                    metrics["accuracy"].make_timeless()

                    log_message_tag = "Widening model (expand_model_weights = {})" \
                        .format(config.expand_model_weights)
                    log_message_value = "Model index: {}, " \
                                        "Previous size {}, " \
                                        "New size {}," \
                                        "step: {}".format(
                        model_idx,
                        previous_layer_size,
                        new_layer_size,
                        step)

                    if config.use_tflogger:
                        logger.log_text(tag=log_message_tag,
                                        value=log_message_value
                                        )
                    logging.info(log_message_tag + ": " + log_message_value)

                    continue
                else:
                    # Could neither complete the task nor is there a scope to grow the model.
                    # No need to check for early stopping anymore
                    metrics["loss"].make_timeless()
                    metrics["accuracy"].make_timeless()
                    # Time to meet the creator
                    should_stop_curriculum = True
                # logging.info("When training model, model_index: {}, early stopping after {} epochs".format(
                #     model_idx, step))
                # logging.info("When training model, model_index: {}, loss = {} for the best performing model".format(
                #     model_idx, metrics["loss"].get_best_so_far()))
                # logging.info(
                #     "When training model, model_index: {}, accuracy = {} for the best performing model".format
                #     (model_idx, metrics["accuracy"].get_best_so_far()))
                # break

    return should_stop_curriculum, model


def evaluate(model, config, data_iterator, tr, logger, device, model_idx, curriculum_idx):
    """Method to evaluate the performance of the model on a particular dataiterator.
    No training takes place in this mode.

    Args:
        model: model object.
        config: config dictionary.
        data_iterator: data iterator object
        tr: training statistics dictionary.
        logger: logger object.
        model_idx: index of the model being evaluated
        curriculum_idx: index of the curriculum which is being used FOR evaluating the current model
    """
    for step in range(config.evaluate_over_n):

        data = data_iterator.next()
        seqloss = 0
        average_accuracy = 0

        model.reset_hidden(batch_size=config.batch_size)

        for i in range(0, data["datalen"]):

            x = torch.from_numpy(numpy.asarray(data['x'][i])).to(device)
            y = torch.from_numpy(numpy.asarray(data['y'][i])).to(device)
            mask = float(data["mask"][i])

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
        tr.updates_done += 1
        if tr.updates_done % 1 == 0:
            logging.info("When evaluating model index: {}, on curriculum index: {}, "
                         "examples seen: {}, running average of BCE: {}, "
                         "average accuracy for last batch: {}, "
                         "running average of accuracy: {}".format(model_idx, curriculum_idx,
                                                                  tr.updates_done * config.batch_size,
                                                                  running_average_bce,
                                                                  average_accuracy,
                                                                  running_average_accuracy))

    if config.use_tflogger:
        logger.log_scalar("eval_avg_loss_model_idx_" + str(model_idx), running_average_bce, curriculum_idx)
        logger.log_scalar("eval_average_accuracy_model_idx_" + str(model_idx), running_average_accuracy, curriculum_idx)


def train_curriculum():
    """Runs the experiment."""

    config = create_config(args.config)

    logging.basicConfig(level=logging.INFO, filename="log.txt", filemode="w")

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

    for model_idx, curriculum_config in enumerate(curriculum_generator(config)):
        logging.info("Starting curriculum index:{}, having sequence length: {}".format(model_idx,
                                                                                       curriculum_config.seq_len))
        data_iterator = get_data_iterator(curriculum_config)
        experiment.register_data_iterator(data_iterator)
        tr = MyContainer()
        tr.updates_done = 0
        tr.average_bce = []
        tr.average_accuracy = []
        experiment.register_train_statistics(tr)
        metrics = get_metric_registry(time_span=curriculum_config.time_span)
        should_stop_curriculum, model = train(experiment, model, curriculum_config, data_iterator, tr, logger, device,
                                              metrics, model_idx)

        model.eval()
        for curriculum_idx, curriculum_config_for_eval in enumerate(curriculum_generator(config)):
            if curriculum_idx >= model_idx + 10:
                break
            tr_for_eval = MyContainer()
            tr_for_eval.updates_done = 0
            tr_for_eval.average_bce = []
            tr_for_eval.average_accuracy = []
            data_iterator_for_eval = get_data_iterator(curriculum_config_for_eval, seed=config.curriculum_seed)
            evaluate(model, curriculum_config_for_eval, data_iterator_for_eval, tr_for_eval,
                     logger, device, model_idx, curriculum_idx)
        model.train()

        if (should_stop_curriculum):
            logging.info("Stopping curriculum after curriculum index:{}, seq_len: {}".format(model_idx,
                                                                                             curriculum_config.seq_len))
            break


if __name__ == '__main__':
    train_curriculum()

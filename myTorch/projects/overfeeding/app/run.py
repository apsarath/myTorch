import argparse
import logging

import numpy
import torch
import torch.nn.functional as F

from myTorch.projects.overfeeding.curriculum.curriculum import curriculum_generator
from myTorch.projects.overfeeding.utils.bootstrap import prepare_experiment, try_to_restart_experiment, get_data_iterator
from myTorch.projects.overfeeding.utils.metric import get_metric_registry
from myTorch.projects.overfeeding.utils.evaluate import evaluate_over_curriculum
from myTorch.utils import MyContainer, create_config, compute_grad_norm
from myTorch.projects.overfeeding.utils.spectral import compute_spectral_properties

parser = argparse.ArgumentParser(description="Algorithm Learning Task")
# parser.add_argument("--config", type=str, default="config/shagun/associative_recall.yaml", help="config file path.")
parser.add_argument("--config", type=str, default="../config/default.yaml", help="config file path.")
parser.add_argument("--force_restart", type=bool, default=False, help="if True start training from scratch.")
args = parser.parse_args()
logging.basicConfig(level=logging.INFO, filename="log.txt", filemode="w")


def train_one_curriculum(experiment, metrics, curriculum_idx):
    """Training loop.

    Args:
        experiment: experiment object.
    """
    config = experiment._config
    should_stop_curriculum = False
    tr = experiment._train_statistics
    data_iterator = experiment._data_iterator
    logger = experiment._logger
    model = experiment._model

    for step in range(tr.updates_done, config.max_steps):
        data = data_iterator.next()
        # seqloss = 0
        # num_correct = 0.0
        # num_total = 0.0

        seqloss, num_correct, num_total = model.train_over_one_data_iterate(data, task=curriculum_idx)
        average_accuracy = num_correct/num_total
        x_shape = data["x"].shape
        average_accuracy /= (x_shape[1] * x_shape[2])
        tr.average_bce.append(seqloss.item())
        tr.average_accuracy.append(average_accuracy)
        running_average_bce = sum(tr.average_bce) / len(tr.average_bce)
        running_average_accuracy = sum(tr.average_accuracy) / len(tr.average_accuracy)


        for param in model.parameters():
            param.grad.clamp_(config.grad_clip[0], config.grad_clip[1])

        model.optimizer.step()

        gradient_norm = compute_grad_norm(parameters=model.parameters()).item()

        if tr.updates_done % 10 == 0:
            if config.use_tflogger:

                if config.log_grad_norm:

                    for key in ['_W_h2o',
                                '_list_of_modules.0._W_x2i',
                                '_list_of_modules.0._W_h2i',
                                '_list_of_modules.0._W_h2f',
                                '_list_of_modules.0._W_h2o',
                                '_list_of_modules.0._W_h2c']:
                        A = model.state_dict()[key].cpu().numpy()
                        spectral_metrics = compute_spectral_properties(A)
                        logger.log_scalar("model_index__{}__condition_number__{}".format(curriculum_idx, key),
                                          spectral_metrics.condition_number, step + 1)
                        logger.log_scalar("model_index__{}__spectral_norm__{}".format(curriculum_idx, key) + str(curriculum_idx),
                                          spectral_metrics.spectral_norm, step + 1)
                        logger.log_scalar("spectral_radius_{}_".format(key) + str(curriculum_idx),
                                          spectral_metrics.spectral_radius, step + 1)

                logger.log_scalar("train_running_avg_loss_model_idx_" + str(curriculum_idx), running_average_bce,
                                  step + 1)
                logger.log_scalar("train_loss_model_idx_" + str(curriculum_idx), tr.average_bce[-1], step + 1)
                logger.log_scalar("train_average accuracy_model_idx_" + str(curriculum_idx), average_accuracy, step + 1)
                logger.log_scalar("train_running_average_accuracy_model_idx_" + str(curriculum_idx),
                                  running_average_accuracy,
                                  step + 1)
                logger.log_scalar("train_gradient_norm_model_idx_" + str(curriculum_idx), gradient_norm,
                                  step + 1)
        metrics["loss"].update(tr.average_bce[-1])
        metrics["accuracy"].update(tr.average_accuracy[-1])

        tr.updates_done += 1
        if tr.updates_done % 1 == 0:

            str_to_log = "When training model, model_index: {}, " \
                         "examples seen: {}, " \
                         "running average of BCE: {}, " \
                         "average accuracy for last batch: {}, " \
                         "running average of accuracy: {},".format(curriculum_idx,
                                                                   tr.updates_done * config.batch_size,
                                                                    running_average_bce,
                                                                    average_accuracy,
                                                                    running_average_accuracy)
            if config.log_grad_norm:
                str_to_log=str_to_log+"gradient norm: {}".format(gradient_norm)
            logging.info(str_to_log)

        if tr.updates_done % config.save_every_n == 0:
            experiment.save()

        # if (metrics["accuracy"].is_best_so_far()):
        #     experiment.save(tag="best")

        # if (tr.updates_done > config.average_over_last_n):
        #     average_accuracy_array = np.asarray(tr.average_accuracy)[-config.average_over_last_n:]
        #     if (np.mean(average_accuracy_array) > 0.8):
        #         # Could complete the task. No need to grow the model
        #         should_stop_curriculum = False
        #         break
        #
        #     elif (metrics["loss"].should_stop_early() or metrics["accuracy"].should_stop_early()):
        #         # Could not complete the task
        #         # Lets try to grow:
        #         if (model.can_make_net_wider(expanded_layer_size=config.expanded_layer_size)):
        #             # Lets expand
        #             previous_layer_size = model.layer_size
        #             experiment.save(tag="model_before_expanding")
        #             model.make_net_wider(expanded_layer_size=config.expanded_layer_size,
        #                                  can_make_optimizer_wider=config.make_optimizer_wider,
        #                                  use_noise=config.use_noise,
        #                                  use_random_noise=config.use_random_noise)
        #             previous_optimizer_state_dict = deepcopy(model.optimizer.state_dict())
        #             new_layer_size = model.layer_size
        #
        #             wider_model = Recurrent(device, config.input_size, config.output_size,
        #                                     num_layers=config.num_layers, layer_size=new_layer_size,
        #                                     cell_name=config.model, activation=config.activation,
        #                                     output_activation="linear")
        #
        #             if (config.expand_model_weights):
        #                 # load the expanded weights
        #                 wider_model.load_state_dict(model.state_dict())
        #
        #             new_config = deepcopy(config)
        #             new_config.lr = new_config.new_lr
        #             optimizer = get_optimizer(wider_model.parameters(), config)
        #
        #             log_message_value = "Model index: {}. " \
        #                                 "Previous learning rate {}. " \
        #                                 "New learning rate {}," \
        #                                 "step: {}".format(
        #                 model_idx,
        #                 config.lr,
        #                 new_config.lr,
        #                 step)
        #
        #             log_message_tag = "New learning rate for the optimizer"
        #
        #             if config.use_tflogger:
        #                 logger.log_text(tag=log_message_tag,
        #                                 value=log_message_value
        #                                 )
        #             logging.info(log_message_tag + ": " + log_message_value)
        #
        #             if (config.make_optimizer_wider):
        #                 # get new optimizer and do the standard bookkeeping tasks
        #                 prev_param_state_values = list(previous_optimizer_state_dict['state'].values())
        #                 param_names_in_new_optimizer = optimizer.state_dict()['param_groups'][0]['params']
        #                 for index, param in enumerate(optimizer.param_groups[0]['params']):
        #                     new_value = prev_param_state_values[index]
        #                     new_value['exp_avg'] = new_value['exp_avg'].to(device)
        #                     new_value['exp_avg_sq'] = new_value['exp_avg_sq'].to(device)
        #                     optimizer.state[0][param_names_in_new_optimizer[index]] = new_value
        #                 log_message_value = "Model index: {}. " \
        #                                     "Previous size {}. " \
        #                                     "New size {}," \
        #                                     "step: {}".format(
        #                     model_idx,
        #                     previous_layer_size,
        #                     new_layer_size,
        #                     step)
        #
        #                 log_message_tag = "Widening Optimizer"
        #
        #                 if config.use_tflogger:
        #                     logger.log_text(tag=log_message_tag,
        #                                     value=log_message_value
        #                                     )
        #                 logging.info(log_message_tag + ": " + log_message_value)
        #
        #             model = wider_model.to(device)
        #             model.register_optimizer(optimizer)
        #             experiment.register_model(model)
        #
        #             experiment.save(tag="model_after_expanding")
        #
        #             # Now we will reset the counters and continue training
        #             metrics["loss"].reset()
        #             metrics["accuracy"].reset()
        #             metrics["loss"].make_timeless()
        #             metrics["accuracy"].make_timeless()
        #
        #             log_message_tag = "Widening model (expand_model_weights = {})" \
        #                 .format(config.expand_model_weights)
        #             log_message_value = "Model index: {}, " \
        #                                 "Previous size {}, " \
        #                                 "New size {}," \
        #                                 "step: {}".format(
        #                 model_idx,
        #                 previous_layer_size,
        #                 new_layer_size,
        #                 step)
        #
        #             if config.use_tflogger:
        #                 logger.log_text(tag=log_message_tag,
        #                                 value=log_message_value
        #                                 )
        #             logging.info(log_message_tag + ": " + log_message_value)
        #
        #             continue
        #         else:
        #             # Could neither complete the task nor is there a scope to grow the model.
        #             # No need to check for early stopping anymore
        #             metrics["loss"].make_timeless()
        #             metrics["accuracy"].make_timeless()
        #             # Time to meet the creator
        #             should_stop_curriculum = True
        #         # logging.info("When training model, model_index: {}, early stopping after {} epochs".format(
        #         #     model_idx, step))
        #         # logging.info("When training model, model_index: {}, loss = {} for the best performing model".format(
        #         #     model_idx, metrics["loss"].get_best_so_far()))
        #         # logging.info(
        #         #     "When training model, model_index: {}, accuracy = {} for the best performing model".format
        #         #     (model_idx, metrics["accuracy"].get_best_so_far()))
        #         # break

    return should_stop_curriculum




def train_curriculums():
    """Runs the experiment."""

    config = create_config(args.config)
    logging.info(config.get())
    experiment = prepare_experiment(config)
    try_to_restart_experiment(experiment, force_restart=args.force_restart)
    should_stop_curriculum = False

    for curriculum_idx, curriculum_config in enumerate(curriculum_generator(config)):
        logging.info("Starting curriculum index:{}, having sequence length: {}".format(curriculum_idx,
                                                                                       curriculum_config.seq_len))
        data_iterator = get_data_iterator(curriculum_config)
        experiment.register_data_iterator(data_iterator)

        tr = MyContainer()
        tr.updates_done = 0
        tr.average_bce = []
        tr.average_accuracy = []
        experiment.register_train_statistics(tr)
        metrics = get_metric_registry(time_span=curriculum_config.time_span)
        should_stop_curriculum = train_one_curriculum(experiment, metrics, curriculum_idx)


        for curriculum_idx_for_eval, curriculum_config_for_eval in enumerate(curriculum_generator(config)):
            if curriculum_idx_for_eval == curriculum_idx:
                # No forward transfer
                break
            tr_for_eval = MyContainer()
            tr_for_eval.updates_done = 0
            tr_for_eval.average_bce = []
            tr_for_eval.average_accuracy = []
            experiment.register_train_statistics(tr_for_eval)
            data_iterator_for_eval = get_data_iterator(curriculum_config_for_eval, seed=config.curriculum_seed)
            evaluate_over_curriculum(experiment, data_iterator_for_eval, curriculum_idx, curriculum_idx_for_eval)
        # model.train()

        # if (should_stop_curriculum):
        #     logging.info("Stopping curriculum after curriculum index:{}, seq_len: {}".format(model_idx,
        #                                                                                      curriculum_config.seq_len))
        #     break


if __name__ == '__main__':
    train_curriculums()

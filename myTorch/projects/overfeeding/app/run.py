import argparse
import logging
import os

import numpy as np

from myTorch.projects.overfeeding.curriculum.curriculum import curriculum_generator
from myTorch.projects.overfeeding.utils.bootstrap import prepare_experiment, \
    try_to_restart_experiment, get_data_iterator, expand_model, prepare_one_curriculum
from myTorch.projects.overfeeding.utils.evaluate import evaluate_over_curriculum
from myTorch.projects.overfeeding.utils.spectral import compute_spectral_properties
from myTorch.utils import MyContainer, create_config, compute_grad_norm
import torch

torch.set_num_threads(1)
print(torch.get_num_threads())

parser = argparse.ArgumentParser(description="Algorithm Learning Task")
# parser.add_argument("--config", type=str, default="config/shagun/associative_recall.yaml", help="config file path.")
parser.add_argument("--config", type=str, default="../config/aaai/extra/128_long.yaml", help="config file path.")
parser.add_argument("--force_restart", type=bool, default=False, help="if True start training from scratch.")
args = parser.parse_args()


# logging.basicConfig(level=logging.INFO, filename="log.txt", filemode="w")


def train_one_curriculum(experiment, curriculum_config, curriculum_idx):
    """Training loop.

    Args:
        experiment: experiment object.
    """
    metrics = prepare_one_curriculum(experiment, curriculum_config)
    config = experiment._config
    should_stop_curriculum = False
    tr = experiment._train_statistics
    data_iterator = experiment._data_iterator
    logger = experiment._logger
    model = experiment._model
    print(experiment._device)

    model_expansion_steps_offset = 0
    # Number of extra training steps to be given to the model because it expanded
    is_curriculum_level_completed = False

    while not is_curriculum_level_completed:
        for step in range(tr.updates_done, config.max_steps + model_expansion_steps_offset):

            if (config.task == "ssmnist"):
                data = data_iterator.next(tag="train")
            else:
                data = data_iterator.next()

            result = model.train_over_one_data_iterate(data, task=curriculum_idx)
            if (config.task == "ssmnist"):
                seqloss, num_correct, num_total, num_elementwise_correct, num_total_correct = result
                average_accuracy = num_correct.float() / num_total
                average_accuracy_element_wise = num_elementwise_correct / num_total_correct
                tr.average_bce.append(seqloss.item())
                tr.average_accuracy.append(average_accuracy)
                tr.average_accuracy_element_wise.append(average_accuracy_element_wise)
                running_average_bce = sum(tr.average_bce) / len(tr.average_bce)
                running_average_accuracy = sum(tr.average_accuracy) / len(tr.average_accuracy)
                running_average_accuracy_element_wise = sum(tr.average_accuracy_element_wise) / len(tr.average_accuracy)

            else:
                seqloss, num_correct, num_total = result
                average_accuracy = num_correct / num_total
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
                            logger.log_scalar("model_index__{}__spectral_norm__{}".format(curriculum_idx, key) +
                                              str(curriculum_idx),
                                              spectral_metrics.spectral_norm, step + 1)
                            logger.log_scalar("spectral_radius_{}_".format(key) + str(curriculum_idx),
                                              spectral_metrics.spectral_radius, step + 1)

                    logger.log_scalar("train_running_avg_loss_model_idx_" + str(curriculum_idx), running_average_bce,
                                      step + 1)
                    logger.log_scalar("train_loss_model_idx_" + str(curriculum_idx), tr.average_bce[-1], step + 1)
                    logger.log_scalar("train_average accuracy_model_idx_" + str(curriculum_idx), average_accuracy,
                                      step + 1)
                    logger.log_scalar("train_running_average_accuracy_model_idx_" + str(curriculum_idx),
                                      running_average_accuracy,
                                      step + 1)
                    logger.log_scalar("train_gradient_norm_model_idx_" + str(curriculum_idx), gradient_norm,
                                      step + 1)
                    if(config.task == "ssmnist"):
                        logger.log_scalar("train_average_element_accuracy_model_idx_" + str(curriculum_idx),
                                      average_accuracy_element_wise, step + 1)
                        logger.log_scalar("train_running_element_average_accuracy_model_idx_" + str(curriculum_idx),
                                      running_average_accuracy_element_wise,
                                      step + 1)

            metrics["loss"].update(tr.average_bce[-1])
            metrics["accuracy"].update(tr.average_accuracy[-1])

            tr.updates_done += 1
            if tr.updates_done % 1 == 0:

                if (config.task == "ssmnist"):
                    str_to_log = "When training model, model_index: {}, " \
                             "examples seen: {}, " \
                             "running average of BCE: {}, " \
                             "average accuracy for last batch: {}, " \
                             "running average of accuracy: {}," \
                             "average elementwise accuracy for last batch: {}, " \
                             "running average of elemenetwise accuracy: {}".format(curriculum_idx,
                                                                               tr.updates_done * config.batch_size,
                                                                               running_average_bce,
                                                                               average_accuracy,
                                                                               running_average_accuracy,
                                                                               average_accuracy_element_wise,
                                                                               running_average_accuracy_element_wise)
                else:
                    str_to_log = "When training model, model_index: {}, " \
                                 "examples seen: {}, " \
                                 "running average of BCE: {}, " \
                                 "average accuracy for last batch: {}, " \
                                 "running average of accuracy: {}".format(curriculum_idx,
                                                                                       tr.updates_done * config.batch_size,
                                                                                       running_average_bce,
                                                                                       average_accuracy,
                                                                                       running_average_accuracy)
                if config.log_grad_norm:
                    str_to_log = str_to_log + "gradient norm: {}".format(gradient_norm)
                logging.info(str_to_log)

            if tr.updates_done % config.save_every_n == 0:
                experiment.save()

            # if (metrics["accuracy"].is_best_so_far()):
            #     experiment.save(tag="best")

        last_n_average_accuracy_array = np.asarray(tr.average_accuracy)[-config.average_over_last_n:]
        if (np.mean(last_n_average_accuracy_array) > config.threshold_accuracy_for_passing_curriculum):
            # Could complete the task. No need to grow the model
            should_stop_curriculum = False
            logging.info("When training model, curriculum_idx: {}, running average loss = {}".format(
                curriculum_idx, running_average_bce))
            logging.info(
                "When training model, curriculum_idx: {}, running average accuracy = {}".format
                (curriculum_idx, running_average_accuracy))
            is_curriculum_level_completed = True
        else:
            # Could not complete the task
            # Lets try to grow:
            experiment, is_expansion_successful = expand_model(experiment,
                                                               config,
                                                               step=tr.updates_done)
            if is_expansion_successful:
                is_curriculum_level_completed = False
                if (config.task == "ssmnist"):
                    data_iterator.reset_iterator()
                model_expansion_steps_offset += 4*config.max_steps
                model = experiment._model
            #     This step seems to be necessary to ensure that model.optimizer is updated. I am not sure why such a
            # dependence exists and would look into later.

            else:
                # Could neither complete the task nor is there a scope to grow the model.
                # Time to meet the creator
                should_stop_curriculum = True
                is_curriculum_level_completed = True

    return should_stop_curriculum


def train_curriculums():
    """Runs the experiment."""

    config = create_config(args.config)
    filename = os.path.join("logs", config.project_name, "{}__{}".format(config.ex_name, "log.txt"))
    # filename = "log.txt"
    logging.basicConfig(level=logging.INFO, filename=filename, filemode="w")
    logging.info(config.get())
    experiment = prepare_experiment(config)
    try_to_restart_experiment(experiment, force_restart=args.force_restart)
    should_stop_curriculum = False
    is_expansion_successful = True

    for curriculum_idx, curriculum_config in enumerate(curriculum_generator(config)):
        logging.info("Starting curriculum index:{}, having sequence length: {}".format(curriculum_idx,
                                                                                       curriculum_config.seq_len))

        if (curriculum_config.stop_if_curriculum_failed):
            should_stop_curriculum = train_one_curriculum(experiment, curriculum_config, curriculum_idx)
        else:
            _ = train_one_curriculum(experiment, curriculum_config, curriculum_idx)
        #
        for curriculum_idx_for_eval, curriculum_config_for_eval in enumerate(curriculum_generator(config)):
            # if curriculum_idx_for_eval == curriculum_idx:
                # No forward transfer
                # break
                # continue
            tr_for_eval = MyContainer()
            tr_for_eval.updates_done = 0
            tr_for_eval.average_bce = []
            tr_for_eval.average_accuracy = []
            tr_for_eval.average_accuracy_element_wise = []
            experiment.register_train_statistics(tr_for_eval)
            data_iterator_for_eval = get_data_iterator(curriculum_config_for_eval, seed=config.curriculum_seed)
            evaluate_over_curriculum(experiment, data_iterator_for_eval, curriculum_idx, curriculum_idx_for_eval)

        if config.expand_model:
            experiment, is_expansion_successful = expand_model(experiment,
                                                               config,
                                                               step=0)
            print("expansion")
            print(is_expansion_successful)
        # model.train()

        if (should_stop_curriculum):
            logging.info("Stopping curriculum after curriculum index:{}, seq_len: {}".format(curriculum_idx,
                                                                                             curriculum_config.seq_len))
            break


if __name__ == '__main__':
    train_curriculums()

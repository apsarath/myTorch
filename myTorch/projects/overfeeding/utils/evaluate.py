import logging


def evaluate_over_curriculum(experiment, data_iterator_for_eval, curriculum_idx, curriculum_eval_idx):
    """Method to evaluate the performance of the model on a particular dataiterator.
    No training takes place in this mode.

    Args:
        experiment: Experiment object
        data_iterator_for_eval: data iterator correspondong to curriculum_eval_idx
        curriculum_idx: index of the curriclum being trained
        curriculum_eval_idx: index of the curriculum which is being used for evaluating the current model
    """

    experiment.eval_mode()
    config = experiment._config
    tr = experiment._train_statistics
    device = experiment._device
    logger = experiment._logger
    model = experiment._model

    for step in range(config.evaluate_over_n):

        if (config.task == "ssmnist"):
            data = data_iterator_for_eval.next(tag="train")
        else:
            data = data_iterator_for_eval.next()
        model.reset_hidden(batch_size=config.batch_size)

        result = model.evaluate_over_one_data_iterate(data, task=curriculum_idx)
        if (config.task == "ssmnist"):
            seqloss, num_correct, num_total, num_elementwise_correct, num_total_correct = result
            average_accuracy = num_correct / num_total
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

        tr.updates_done += 1
        if tr.updates_done % 1 == 0:
            logging.info("When evaluating curriculum index: {}, on curriculum index: {}, " \
                         "examples seen: {}, running average of BCE: {}, " \
                         "average accuracy for last batch: {}, " \
                         "running average of accuracy: {}, " \
                         "average elementwise accuracy for last batch: {}, "
                         "running average of elemenetwise accuracy: {}".format(curriculum_idx, curriculum_eval_idx,
                                                                               tr.updates_done * config.batch_size,
                                                                               running_average_bce,
                                                                               average_accuracy,
                                                                               running_average_accuracy,
                                                                               average_accuracy_element_wise,
                                                                               running_average_accuracy_element_wise))

    if config.use_tflogger:
        logger.log_scalar("eval_avg_loss_for_curriculum_idx" + str(curriculum_idx), running_average_bce,
                          curriculum_eval_idx)
        logger.log_scalar("eval_average_accuracy_model_for_curriculum_idx" + str(curriculum_idx),
                          running_average_accuracy, curriculum_eval_idx)
        logger.log_scalar("eval_average_elementwise_accuracy_model_for_curriculum_idx" + str(curriculum_idx),
                          running_average_accuracy_element_wise, curriculum_eval_idx)

    experiment.train_mode()

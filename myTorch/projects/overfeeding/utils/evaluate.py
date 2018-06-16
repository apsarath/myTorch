import logging

import numpy
import torch
import torch.nn.functional as F

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

        data = data_iterator_for_eval.next()
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
            logging.info("When evaluating curriculum index: {}, on curriculum index: {}, "
                         "examples seen: {}, running average of BCE: {}, "
                         "average accuracy for last batch: {}, "
                         "running average of accuracy: {}".format(curriculum_idx, curriculum_eval_idx,
                                                                  tr.updates_done * config.batch_size,
                                                                  running_average_bce,
                                                                  average_accuracy,
                                                                  running_average_accuracy))

    if config.use_tflogger:
        logger.log_scalar("eval_avg_loss_for_curriculum_idx" + str(curriculum_idx), running_average_bce,
                          curriculum_eval_idx)
        logger.log_scalar("eval_average_accuracy_model_for_curriculum_idx" + str(curriculum_idx),
                          running_average_accuracy, curriculum_eval_idx)

    experiment.train_mode()

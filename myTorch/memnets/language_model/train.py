import argparse
import hashlib
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from myTorch import Experiment
from myTorch.memnets.language_model import data
from myTorch.memnets.language_model.lm import LanguageModel
from myTorch.memnets.language_model.util import make_model_wider
from myTorch.utils import MyContainer, get_optimizer, create_config
# from myTorch.memnets.recurrent_net import Recurrent
# from myTorch.task.copy_task import CopyData
# from myTorch.task.repeat_copy_task import RepeatCopyData
# from myTorch.task.associative_recall_task import AssociativeRecallData
# from myTorch.task.copying_memory import CopyingMemoryData
# from myTorch.task.adding_task import AddingData
# from myTorch.task.denoising import DenoisingData
from myTorch.utils.logging import Logger

parser = argparse.ArgumentParser(description="Algorithm Learning Task")
parser.add_argument("--config", type=str,
                    default="/Users/shagun/projects/myTorch/myTorch/memnets/language_model/ptb.yaml",
                    help="config file path.")
parser.add_argument("--force_restart", type=bool, default=False, help="if True start training from scratch.")
parser.add_argument("--log_file_path", type=str, default="log.txt", help="log file path.")


def _safe_exp(x):
    try:
        return math.exp(x)
    except:
        return 0.0


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data


def get_batch(source, i, bptt, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len]
    done = False
    if i + seq_len > source.shape[0] or data.shape[0] < bptt:
        done = True
    return data, target, done, i + seq_len


def get_batched_data(config):
    fn = 'corpus.{}.data'.format(hashlib.md5(config.data.encode()).hexdigest())
    if os.path.exists(fn):
        print('Loading cached dataset...')
        corpus = torch.load(fn)
    else:
        print('Producing dataset...')
        corpus = data.Corpus(config.data)
        torch.save(corpus, fn)

    batched_data = {}
    batched_data["train"] = batchify(corpus.train, config.batch_size)
    batched_data["valid"] = batchify(corpus.valid, config.eval_batch_size)
    batched_data["test"] = batchify(corpus.test, config.test_batch_size)
    vocab = corpus.dictionary
    print("Vocab size : {}".format(len(vocab)))
    return batched_data, vocab


def run_epoch(epoch_id, mode, experiment, model, config, batched_data, tr, logger, device,
              prefix="before_expansion_"):
    """Training loop.

    Args:
        experiment: experiment object.
        model: model object.
        config: config dictionary.
        data_iterator: data iterator object
        tr: training statistics dictionary.
        logger: logger object.
    """

    assert (mode == "train" or mode == "test" or mode == "valid")
    if mode == "train":
        batch_size = config.batch_size
    elif mode == "valid":
        batch_size = config.eval_batch_size
    elif mode == "test":
        batch_size = config.test_batch_size

    batched_data[mode] = batched_data[mode].to(device)

    model.reset_hidden(batch_size=batch_size)
    num_total_words = batched_data[mode].shape[0] * batched_data[mode].shape[1]
    done = False
    step = 0
    curr_epoch_loss = []
    curr_epoch_acc_at_k = {1: [], 2: [], 3: [], 5: []}
    start_time = time.time()
    while not done:
        model.repackage_hidden()

        x, y, done, tr.mini_batch_id[mode] = get_batch(batched_data[mode], tr.mini_batch_id[mode], config.bptt)
        seqloss = 0
        output_logits = model(x)

        curr_time_steps = y.shape[0]
        for i in range(curr_time_steps):
            seqloss += F.cross_entropy(output_logits[i], y[i])
        seqloss /= curr_time_steps

        # accuracy computation
        def _acc_at_k(k):
            _, ids = torch.topk(torch.stack(output_logits, dim=0), k, dim=2)
            eq_vec = torch.eq(y.unsqueeze(-1).expand_as(ids), ids).double()
            acc = torch.mean(torch.sum(eq_vec, dim=-1)).cpu().item() * 100.0
            return acc

        for k in curr_epoch_acc_at_k:
            curr_epoch_acc_at_k[k].append(_acc_at_k(k))

        tr.average_loss[mode].append(seqloss.item())
        curr_epoch_loss.append(seqloss.item())

        running_average = sum(tr.average_loss[mode]) / len(tr.average_loss[mode])

        if config.use_tflogger and mode == "train":
            logger.log_scalar(prefix + "running_avg_loss", running_average, tr.updates_done[mode] + 1)
            logger.log_scalar(prefix + "loss", tr.average_loss[mode][-1], tr.updates_done[mode] + 1)
            logger.log_scalar(prefix + "running_perplexity", _safe_exp(running_average), tr.updates_done[mode] + 1)
            logger.log_scalar(prefix + "inst_perplexity", _safe_exp(tr.average_loss[mode][-1]),
                              tr.updates_done[mode] + 1)

        if mode == "train":
            model.optimizer.zero_grad()
            seqloss.backward(retain_graph=False)
            torch.nn.utils.clip_grad_norm(model.parameters(), config.grad_clip_norm)
            model.optimizer.step()

        tr.updates_done[mode] += 1
        step += 1
        if tr.updates_done[mode] % 1e6 == 0 and mode == "train":
            logging.info(prefix[:-1] + " Epoch : {}, {} %: {}, step : {}, time : {}".format(epoch_id, mode, (
                    100.0 * step * batch_size * curr_time_steps / num_total_words), tr.updates_done[mode],
                                                                                            time.time() - start_time))
            logging.info(prefix[:-1] + " inst loss: {}, inst perp: {}".format(tr.average_loss[mode][-1],
                                                                              _safe_exp(tr.average_loss[mode][-1])))

    curr_epoch_avg_loss = np.mean(np.array(curr_epoch_loss))
    tr.average_loss_per_epoch[mode].append(curr_epoch_avg_loss)
    for k in curr_epoch_acc_at_k:
        curr_epoch_acc_at_k[k] = np.mean(np.array(curr_epoch_acc_at_k[k]))
    tr.acc_at_k_per_epoch[mode].append(curr_epoch_acc_at_k)

    logging.info(prefix[:-1] + "Avg {} loss: {}, BPC : {}, Avg perp: {}, time : {}".format(mode, curr_epoch_avg_loss,
                                                                                           curr_epoch_avg_loss / 0.693,
                                                                                           _safe_exp(
                                                                                               curr_epoch_avg_loss),
                                                                                           time.time() - start_time))

    for k in curr_epoch_acc_at_k:
        logging.info(prefix[:-1] + "Acc at {} : {}".format(k, curr_epoch_acc_at_k[k]))

    batched_data[mode] = batched_data[mode].to("cpu")

    if mode != "train":
        logger.log_scalar(prefix + "loss_{}".format(mode), curr_epoch_avg_loss, epoch_id + 1)
        logger.log_scalar(prefix + "perplexity_{}".format(mode), _safe_exp(curr_epoch_avg_loss), epoch_id + 1)


def create_experiment(config, expand_model=False, model=None):
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

    batch_data, vocab = get_batched_data(config)

    if expand_model:
        #     Need to expand the given model
        if model.can_make_net_wider(expanded_layer_size=config.expanded_layer_size):
            experiment.save(tag="model_before_expanding")
            model_idx = 1
            step = 0
            model, optimizer = make_model_wider(model, config, device, model_idx, step, logger, logging, vocab)
            model.register_optimizer(optimizer)
            experiment.register_model(model)
            experiment.save(tag="model_after_expanding")

    if not model:
        model = LanguageModel(device, len(vocab), config.input_emb_size,
                              num_layers=config.num_layers, layer_size=config.layer_size,
                              cell_name=config.model, activation=config.activation,
                              output_activation="linear", layer_norm=config.layer_norm,
                              identity_init=config.identity_init, chrono_init=config.chrono_init,
                              t_max=config.bptt / 3, memory_size=config.memory_size, k=config.k,
                              use_relu=config.use_relu).to(device)
    experiment.register_model(model)

    optimizer = get_optimizer(model.parameters(), config)
    model.register_optimizer(optimizer)

    tr = MyContainer()

    tr.mini_batch_id, tr.updates_done, tr.average_loss, tr.average_loss_per_epoch = {}, {}, {}, {}
    tr.acc_at_k_per_epoch = {}

    for mode in ["train", "valid", "test"]:
        tr.mini_batch_id[mode] = 0
        tr.updates_done[mode] = 0
        tr.average_loss[mode] = []
        tr.average_loss_per_epoch[mode] = []
        tr.acc_at_k_per_epoch[mode] = []

    experiment.register_train_statistics(tr)

    return experiment, model, batch_data, tr, logger, device


def run_experiment(args):
    """Runs the experiment."""

    config = create_config(args.config)

    logging.info(config.get())

    experiment, model, batched_data, tr, logger, device = create_experiment(config)

    if not args.force_restart:
        if experiment.is_resumable():
            experiment.resume()
    else:
        experiment.force_restart()

    for i in range(config.num_epochs):
        logging.info("\n#####################\n Epoch id: {}\n".format(i + 1))
        for mode in ["train", "valid", "test"]:
            tr.mini_batch_id[mode] = 0
            run_epoch(i, mode, experiment, model, config, batched_data, tr, logger, device)

    logging.info("\n#####################\n Best Model\n")
    min_id = np.argmin(np.array(tr.average_loss_per_epoch["valid"]))
    valid_loss = tr.average_loss_per_epoch["valid"][min_id] / 0.693
    logging.info("Best Valid BPC : {}, perplexity : {}".format(valid_loss, _safe_exp(valid_loss)))
    test_loss = tr.average_loss_per_epoch["test"][min_id] / 0.693
    logging.info("Best Test BPC : {}, perplexity : {}".format(test_loss, _safe_exp(test_loss)))
    for k in tr.acc_at_k_per_epoch["test"][min_id]:
        logging.info("Acc at {} : {}".format(k, tr.acc_at_k_per_epoch["test"][min_id][k]))


def run_experiment_before_expansion(args):
    """Runs the experiment before expanding the model"""

    config = create_config(args.config)
    prefix = "before_expansion_"

    logging.info(config.get())

    experiment, model, batched_data, tr, logger, device = create_experiment(config,
                                                                            expand_model=False,
                                                                            model=None)

    if not args.force_restart:
        if experiment.is_resumable():
            experiment.resume()
    else:
        experiment.force_restart()

    for i in range(config.num_epochs):
        logging.info("\n#####################\n {} Epoch id: {}\n".format(prefix[:-1], i + 1))
        print("\n#####################\n {} Epoch id: {}\n".format(prefix[:-1], i + 1))
        for mode in ["train", "valid", "test"]:
            tr.mini_batch_id[mode] = 0
            run_epoch(i, mode, experiment, model, config, batched_data, tr, logger, device, prefix)

    logging.info("\n#####################\n Best Model {}\n".format(prefix[:-1]))
    min_id = np.argmin(np.array(tr.average_loss_per_epoch["valid"]))
    valid_loss = tr.average_loss_per_epoch["valid"][min_id] / 0.693
    logging.info("{} Best Valid BPC : {}, perplexity : {}".format(prefix[:-1], valid_loss, _safe_exp(valid_loss)))
    test_loss = tr.average_loss_per_epoch["test"][min_id] / 0.693
    logging.info("{} Best Test BPC : {}, perplexity : {}".format(prefix[:-1], test_loss, _safe_exp(test_loss)))
    for k in tr.acc_at_k_per_epoch["test"][min_id]:
        logging.info("{} Acc at {} : {}".format(prefix[:-1], k, tr.acc_at_k_per_epoch["test"][min_id][k]))
    return config, model


def run_experiment_after_expansion(config, model):
    # expand model
    # config = create_config(args.config)
    prefix = "after_expansion_"

    logging.info(config.get())

    experiment, model, batched_data, tr, logger, device = create_experiment(config,
                                                                            expand_model=True,
                                                                            model=model)
    print(config.num_epochs)
    print("successfully expnaded the model")
    for i in range(config.num_epochs):

        logging.info("\n#####################\n {} Epoch id: {}\n".format(prefix[:-1], i + 1))
        print("\n#####################\n {} Epoch id: {}\n".format(prefix[:-1], i + 1))
        for mode in ["train", "valid", "test"]:
            tr.mini_batch_id[mode] = 0
            run_epoch(i, mode, experiment, model, config, batched_data, tr, logger, device, prefix)

    logging.info("\n#####################\n Best Model {}\n".format(prefix[:-1]))
    min_id = np.argmin(np.array(tr.average_loss_per_epoch["valid"]))
    valid_loss = tr.average_loss_per_epoch["valid"][min_id] / 0.693
    logging.info("{} Best Valid BPC : {}, perplexity : {}".format(prefix[:-1], valid_loss, _safe_exp(valid_loss)))
    test_loss = tr.average_loss_per_epoch["test"][min_id] / 0.693
    logging.info("{} Best Test BPC : {}, perplexity : {}".format(prefix[:-1], test_loss, _safe_exp(test_loss)))
    for k in tr.acc_at_k_per_epoch["test"][min_id]:
        logging.info("{} Acc at {} : {}".format(prefix[:-1], k, tr.acc_at_k_per_epoch["test"][min_id][k]))


if __name__ == '__main__':
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        handlers=[
                            logging.FileHandler(args.log_file_path),
                            logging.StreamHandler()
                        ]
                        )

    config, model = run_experiment_before_expansion(args)
    run_experiment_after_expansion(config, model)

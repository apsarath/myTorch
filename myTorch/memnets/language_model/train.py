import numpy
import math
import argparse
import logging
import os
import hashlib

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
from myTorch.memnets.language_model import data
from myTorch.memnets.language_model.lm import LanguageModel

import torch.nn.functional as F

parser = argparse.ArgumentParser(description="Algorithm Learning Task")
parser.add_argument("--config", type=str, default="config/default.yaml", help="config file path.")
parser.add_argument("--force_restart", type=bool, default=False, help="if True start training from scratch.")

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
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len]
    done = False
    if i+seq_len > source.shape[0]:
        done = True 
    return data, target, done, i+seq_len

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
    return batched_data, vocab



def run_epoch(epoch_id, mode, experiment, model, config, batched_data, tr, logger, device):
    """Training loop.

    Args:
        experiment: experiment object.
        model: model object.
        config: config dictionary.
        data_iterator: data iterator object
        tr: training statistics dictionary.
        logger: logger object.
    """

    assert(mode == "train" or mode == "test" or mode == "valid")
    if mode == "train":
        batch_size = config.batch_size
    elif mode == "valid":
        batch_size = config.eval_batch_size
    elif mode == "test":
        batch_size = config.test_batch_size

    model.reset_hidden(batch_size=batch_size)
    num_total_words = batched_data[mode].shape[0] * batched_data[mode].shape[1]
    done = False
    step = 0
    while not done:
        if config.inter_saving is not None:
            if tr.updates_done[mode] in config.inter_saving and mode == "train":
                experiment.save(str(tr.updates_done[mode]))

        if mode == "train":
            model.repackage_hidden()

        x, y, done, tr.mini_batch_id[mode] = get_batch(batched_data[mode], tr.mini_batch_id[mode], config.bptt)
        x, y = x.to(device), y.to(device)
        seqloss = 0
        output_logits = model(x)

        for i in range(config.bptt):
            seqloss += F.cross_entropy(output_logits[i], y[i])
        seqloss /= config.bptt
        
        tr.average_loss[mode].append(seqloss.item())

        running_average = sum(tr.average_loss[mode]) / len(tr.average_loss[mode])

        if config.use_tflogger:
            logger.log_scalar("running_avg_loss", running_average, step + 1)
            logger.log_scalar("loss", tr.average_loss[mode][-1], step + 1)
            logger.log_scalar("running_perplexity", math.exp(running_average), step + 1)
            logger.log_scalar("inst_perplexity", math.exp(tr.average_loss[mode][-1]), step + 1)

        if mode == "train":
            model.optimizer.zero_grad()
            seqloss.backward(retain_graph=False)
            torch.nn.utils.clip_grad_norm(model.parameters(), config.grad_clip_norm)
            model.optimizer.step()

        tr.updates_done[mode] +=1
        step += 1
        if tr.updates_done[mode] % 1 == 0:
            logging.info("Epoch : {}, {} %: {}".format(epoch_id, mode, (100.0*step*batch_size*config.bptt/num_total_words)))
            logging.info("inst loss: {}, avg perp: {}".format(tr.average_loss[mode][-1], math.exp(running_average)))
            
        if tr.updates_done[mode] % config.save_every_n == 0 and mode == "train":
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

    batch_data, vocab = get_batched_data(config)
    
    model = LanguageModel(device, len(vocab), config.input_emb_size,
                      num_layers=config.num_layers, layer_size=config.layer_size,
                      cell_name=config.model, activation=config.activation,
                      output_activation="linear", layer_norm=config.layer_norm,
                      identity_init=config.identity_init, chrono_init=config.chrono_init,
                      t_max=config.t_max).to(device)
    experiment.register_model(model)

    optimizer = get_optimizer(model.parameters(), config)
    model.register_optimizer(optimizer)

    tr = MyContainer()

    tr.mini_batch_id, tr.updates_done, tr.average_loss = {}, {}, {}

    for mode in ["train", "valid", "test"]:
        tr.mini_batch_id[mode] = 0
        tr.updates_done[mode] = 0
        tr.average_loss[mode] = []
        

    experiment.register_train_statistics(tr)

    return experiment, model, batch_data, tr, logger, device


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

    for i in range(config.num_epochs):
        for mode in ["train", "valid", "test"]:
            tr.mini_batch_id[mode] = 0
            if mode != "train":
                tr.updates_done[mode] = 0
                tr.average_loss[mode] = []
            run_epoch(i, mode, experiment, model, config, data_iterator, tr, logger, device)


if __name__ == '__main__':
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    run_experiment(args)

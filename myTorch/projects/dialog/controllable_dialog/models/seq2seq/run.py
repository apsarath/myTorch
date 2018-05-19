#!/usr/bin/env python
import numpy as np
import argparse
import logging
import torch
import hashlib
import os
import time
import math

import myTorch
from myTorch.utils import MyContainer, get_optimizer, create_config
from myTorch.utils.logging import Logger
from myTorch.utils.gen_experiment import GenExperiment

from myTorch.projects.dialog.controllable_dialog.data_readers.data_reader import Reader
from myTorch.projects.dialog.controllable_dialog.data_readers.opus import OPUS

from myTorch.projects.dialog.controllable_dialog.models.seq2seq.seq2seq import Seq2Seq

parser = argparse.ArgumentParser(description="seq2seq")
parser.add_argument("--config", type=str, default="config/opus/default.yaml", help="config file path.")
parser.add_argument("--force_restart", type=bool, default=False, help="if True start training from scratch.")

def _safe_exp(x):
    try:
        return math.exp(x)
    except:
        return 0.0

def get_dataset(config):
    hash_string = "{}_{}".format(config.base_data_path, config.num_dialogs)
    fn = 'corpus.{}.data'.format(hashlib.md5(hash_string.encode()).hexdigest())
    fn = os.path.join(config.base_data_path, str(config.num_dialogs), fn)
    if 0:#os.path.exists(fn):
        print('Loading cached dataset...')
        corpus = torch.load(fn)
    else:
        print('Producing dataset...')
        corpus = OPUS(config)
        #torch.save(corpus, fn)

    return corpus

def create_experiment(config):
    device = torch.device(config.device)
    logging.info("using {}".format(config.device))

    experiment = GenExperiment(config.name, config.save_dir)
    experiment.register(tag="config", obj=config)

    logger=None
    if config.use_tflogger:
        logger = Logger(config.tflog_dir)
        experiment.register("logger", logger)

    torch.manual_seed(config.rseed)
        
    corpus = get_dataset(config)
    reader = Reader(config, corpus)

    model = Seq2Seq(config.emb_size_src, len(corpus.str_to_id), config.hidden_dim_src, config.hidden_dim_tgt,
                    corpus.str_to_id[config.pad], bidirectional=config.bidirectional,
                    nlayers_src=config.nlayers_src, nlayers_tgt=config.nlayers_tgt,
                    dropout_rate=config.dropout_rate).to(device)
    logging.info("Num params : {}".format(model.num_parameters))

    experiment.register("model", model)

    optimizer = get_optimizer(model.parameters(), config)
    model.register_optimizer(optimizer)

    tr = MyContainer()

    tr.mini_batch_id, tr.loss_per_epoch = {}, {}
    tr.epoch_id = 0

    for mode in ["train", "valid", "test"]:
        tr.mini_batch_id[mode] = 0
        tr.loss_per_epoch[mode] = []

    experiment.register("train_statistics", tr)

    return experiment, model, reader, tr, logger, device

def run_epoch(epoch_id, mode, experiment, model, config, data_reader, tr, logger, device):
    
    itr = data_reader.itr_generator(mode, tr.mini_batch_id[mode])
    weight_mask = torch.ones(len(data_reader.corpus.str_to_id)).to(device)
    weight_mask[data_reader.corpus.str_to_id[config.pad]] = 0
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight_mask)

    start_time = time.time()
    num_batches = 0
    loss_per_epoch = []
    for mini_batch in itr:
        num_batches = mini_batch["num_batches"]
        model.zero_grad()
        output_logits = model(
                        mini_batch["sources"].to(device), 
                        mini_batch["sources_len"].to(device),
                        mini_batch["targets_input"].to(device),
                        is_training=True if mode=="train" else False)
        loss = loss_fn( output_logits.contiguous().view(-1, output_logits.size(2)), 
                mini_batch["targets_output"].to(device).contiguous().view(-1))
        
        loss_per_epoch.append(loss.item())

        if mode == "train":
            model.optimizer.zero_grad()
            loss.backward(retain_graph=False)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            model.optimizer.step()

        tr.mini_batch_id[mode] += 1

        if tr.mini_batch_id[mode] % 1e6 == 0 and mode == "train":
            logging.info("Epoch : {}, {} %: {}, time : {}".format(epoch_id, mode, (100.0*tr.mini_batch_id[mode]/num_batches), time.time()-start_time))
            logging.info("Running loss: {}, perp: {}".format(running_average, _safe_exp(running_average)))

    if mode == "train":
        experiment.save("current")

    avg_loss = np.mean(np.array(loss_per_epoch))
    tr.loss_per_epoch[mode].append(avg_loss)
        
    logging.info("\n*****************************\n")
    logging.info("{}: loss: {},  perp: {}".format(mode, avg_loss, _safe_exp(avg_loss)))
    logging.info("\n*****************************\n")

    if mode == "valid" and len(tr.loss_per_epoch[mode]) > 1:
        if tr.loss_per_epoch[mode][-1] < np.min(np.array(tr.loss_per_epoch[mode][:-1])):
            logging.info("Saving Best model : loss : {}".format(tr.loss_per_epoch[mode][-1]))
            experiment.save("best_model", "model")


def run_experiment(args):
    """Runs the experiment."""

    config = create_config(args.config)

    logging.info(config.get())

    experiment, model, data_reader, tr, logger, device = create_experiment(config)

    if not args.force_restart:
        if experiment.is_resumable("current"):
            experiment.resume("current")
    else:
        experiment.force_restart("current")
        experiment.force_restart("best_model")

    for i in range(tr.epoch_id, config.num_epochs):
        for mode in ["train", "valid"]:
            tr.mini_batch_id[mode] = 0
            tr.epoch_id = i
            run_epoch(i, mode, experiment, model, config, data_reader, tr, logger, device)
        
if __name__ == '__main__':
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run_experiment(args)

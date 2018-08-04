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
from myTorch.projects.dialog.controllable_dialog.data_readers.cornell_corpus import Cornell
from myTorch.projects.dialog.controllable_dialog.data_readers.twitter_corpus import Twitter

from myTorch.projects.dialog.controllable_dialog.models.lm.lm import LM
from myTorch.projects.dialog.controllable_dialog.models import eval_metrics
from myTorch.projects.dialog.controllable_dialog.models.lm.beam_search import SequenceGenerator

parser = argparse.ArgumentParser(description="lm")
parser.add_argument("--config", type=str, default="config/opus/default.yaml", help="config file path.")

def _safe_exp(x):
    try:
        return math.exp(x)
    except:
        return 0.0

def get_dataset(config):
    if config.dataset == "opus":
        corpus = OPUS(config)
    elif config.dataset == "cornell":
        corpus = Cornell(config)
    elif config.dataset == "twitter":
        corpus = Twitter(config)
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

    model = LM(config.emb_size_src, len(corpus.str_to_id), config.hidden_dim_src, 
                    corpus.str_to_id[config.pad], bidirectional=config.bidirectional,
                    nlayers_src=config.nlayers_src, dropout_rate=config.dropout_rate, 
                    device=device, pretrained_embeddings=corpus.pretrained_embeddings).to(device)

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

def run_epoch(experiment, model, config, data_reader, tr, logger, device):
    
    weight_mask = torch.ones(len(data_reader.corpus.str_to_id)).to(device)
    weight_mask[data_reader.corpus.str_to_id[config.pad]] = 0

    start_time = time.time()
    num_batches = 0
    filename_s = "samples_{}.txt".format(config.ex_name)
    f_s = open(filename_s, "w")
    count = 1

    SeqGen = SequenceGenerator(
            model.decode_step,
            data_reader.corpus.str_to_id[config.eou],
            beam_size=10,
            max_sequence_length=20,
            get_attention=False,
            length_normalization_factor=5.0,
            length_normalization_const=1.0)
    go_id = data_reader.corpus.str_to_id[config.go]
    initial_decoder_input = [[go_id] for _ in range(config.batch_size)]

    while True:
        print("Count : {}".format(count))
        count += 1
        if count > 10:
            break
        model.zero_grad()
        seqs = SeqGen.beam_search(initial_input=initial_decoder_input)
        
        samples = []
        for seq_list in seqs:
            sample = []
            for seq in seq_list:
                for w_id in seq.output:
                    sample.append(data_reader.corpus.id_to_str[w_id])
                sample.append("--- | ---")
            samples.append(sample)

        for sample in samples:
            f_s.write("{}\n".format(" ".join(sample)))

    f_s.close()

    d1, d2 = eval_metrics.distinct_scores(filename_s)
    print("Distinct 1 : {}".format(d1))
    print("Distinct 2 : {}".format(d2))

def run_experiment(args):
    """Runs the experiment."""

    config = create_config(args.config)

    logging.info(config.get())

    experiment, model, data_reader, tr, logger, device = create_experiment(config)

    experiment.resume("best_model", "model")

    run_epoch(experiment, model, config, data_reader, tr, logger, device)

        
if __name__ == '__main__':
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run_experiment(args)

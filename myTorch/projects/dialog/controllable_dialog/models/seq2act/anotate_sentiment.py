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

from myTorch.projects.dialog.controllable_dialog.data_readers.seq2act_data_reader import Reader
from myTorch.projects.dialog.controllable_dialog.data_readers.opus import OPUS
from myTorch.projects.dialog.controllable_dialog.models.seq2act.sentiment_classifier import SentimentClassifier

parser = argparse.ArgumentParser(description="sentiment")
parser.add_argument("--config", type=str, default="config/opus/default.yaml", help="config file path.")
parser.add_argument("--force_restart", type=bool, default=False, help="if True start training from scratch.")

def _safe_exp(x):
    try:
        return math.exp(x)
    except:
        return 0.0

def get_dataset(config):
    if config.dataset == "opus":
        corpus = OPUS(config)
        return corpus


def anotate(config, experiment, opus_corpus, logger, device):
    
    opus_data = opus_corpus.raw_data
    start_time = time.time()

    sentiment_classifier = SentimentClassifier()

    # convert opus_data text to target data.
    def _convert_data(source_data, source_id_to_str):
        target_data = []
        for source_text in source_data.numpy():
            target_text = " ".join([source_id_to_str[w_id] for w_id in source_text])
            target_data.append(target_text)

        return target_data
    
    opus_sources = _convert_data(opus_data["sources"], opus_corpus.id_to_str)
    opus_targets = _convert_data(opus_data["targets_output"], opus_corpus.id_to_str)

    acts = { "source" : [], "target" : []}
    
    print("Anotating sources...")
    for opus_source in opus_sources:
        sentiment = sentiment_classifier.get_sentiment_id(opus_source)
        acts["source"].append(sentiment)
    print("Done.. {}".format(time.time()-start_time))

    for opus_target in opus_targets:
        sentiment = sentiment_classifier.get_sentiment_id(opus_target)
        acts["target"].append(sentiment)
    print("Done targets.. {}".format(time.time()-start_time))

    opus_corpus.save_acts("{}_{}".format("sentiment", len(sentiment_classifier.sentiment_types)), acts)
    

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

    opus_corpus = get_dataset(config)

    return experiment, opus_corpus, logger, device


def run_experiment(args):
    """Runs the experiment."""

    config = create_config(args.config)

    logging.info(config.get())

    experiment, opus_corpus, logger, device = create_experiment(config)

    anotate(config, experiment, opus_corpus, logger, device)
    

        
if __name__ == '__main__':
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run_experiment(args)

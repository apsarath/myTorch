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
from myTorch.projects.dialog.controllable_dialog.data_readers.switchboard_corpus import SwitchBoard
from myTorch.projects.dialog.controllable_dialog.data_readers.frames_corpus import Frames
from myTorch.projects.dialog.controllable_dialog.models.seq2act.seq2act import Seq2Act

parser = argparse.ArgumentParser(description="seq2act")
parser.add_argument("--config", type=str, default="config/opus/default.yaml", help="config file path.")
parser.add_argument("--opus_config", type=str, default="config/opus/default.yaml", help="config file path.")
parser.add_argument("--force_restart", type=bool, default=False, help="if True start training from scratch.")

def _safe_exp(x):
    try:
        return math.exp(x)
    except:
        return 0.0

def get_dataset(config):
    if config.dataset == "switchboard":
        corpus = SwitchBoard(config)
    if config.dataset == "frames":
        corpus = Frames(config)
    elif config.dataset == "opus":
        corpus = OPUS(config)
    return corpus


def anotate(config, experiment, opus_corpus, tr, logger, device):
    
    opus_data = opus_corpus.raw_data
    start_time = time.time()

    acts = { "source" : [], "target" : []}
    
    print("Anotating...")
    for i in range(opus_data["sources"].shape[0]):
        acts["source"].append(np.random.randint(0,11))
        acts["target"].append(np.random.randint(0,11))
    print("Done.. {}".format(time.time()-start_time))

    opus_corpus.save_acts("random", acts)
    

def create_experiment(config, opus_config):
    device = torch.device(config.device)
    logging.info("using {}".format(config.device))

    experiment = GenExperiment(config.name, config.save_dir)
    experiment.register(tag="config", obj=config)

    logger=None
    if config.use_tflogger:
        logger = Logger(config.tflog_dir)
        experiment.register("logger", logger)

    torch.manual_seed(config.rseed)

    opus_corpus = get_dataset(opus_config)

    tr = MyContainer()

    experiment.register("train_statistics", tr)
    return experiment, opus_corpus, tr, logger, device


def run_experiment(args):
    """Runs the experiment."""

    config = create_config(args.config)
    opus_config = create_config(args.opus_config)

    logging.info(config.get())

    experiment, opus_corpus, tr, logger, device = create_experiment(config, opus_config)

    anotate(config, experiment, opus_corpus, tr, logger, device)
    

        
if __name__ == '__main__':
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run_experiment(args)

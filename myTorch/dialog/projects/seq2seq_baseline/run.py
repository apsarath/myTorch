#!/usr/bin/env python

import os
import math
import numpy as np
import argparse
import torch
import myTorch
from myTorch.dialog.datasets import make_dataset

parser = argparse.ArgumentParser(description="Dialog agent Training and Inference")
parser.add_argument('--config', type=str, default="opus", help="config name")
parser.add_argument('--base_dir', type=str, default=None, help="base directory")
parser.add_argument('--config_params', type=str, default="default", help="config params to change")
parser.add_argument('--exp_desc', type=str, default="default", help="additional desc of exp")
parser.add_argument('--mode', type=str, default="train", help="Modes: train, sample")
args = parser.parse_args()


def run():
    assert(args.base_dir)
    #import pdb; pdb.set_trace()
    config = eval(args.config)()

    if args.config_params != "default":
        modify_config_params(config, args.config_params)

    train_dir = os.path.join(args.base_dir, config.train_dir, config.exp_name, config.env_name,
        "{}__{}".format(args.config_params, args.exp_desc))

    logger_dir = os.path.join(args.base_dir, config.logger_dir, config.exp_name, config.env_name,
        "{}__{}".format(args.config_params, args.exp_desc))


    experiment = RLExperiment(config.exp_name, train_dir, config.backup_logger)
    experiment.register_config(config)

    torch.manual_seed(config.seed)
    numpy_rng = np.random.RandomState(seed=config.seed)

    dataset = make_dataset(config.dataset_name, config)


    # seq2seq network init


    # dialog agent init
    if args.mode == "train":
        train_dialog_agent()
    elif args.mode == "sample":
        sample_dialog_agent()


    # trainer init


def train_dialog_agent():
    #train loop
    return

def sample_dialog_agent():
    return

if __name__=="__main__":
    run()

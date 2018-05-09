#!/usr/bin/env python
import numpy
import argparse
import logging
import torch



parser = argparse.ArgumentParser(description="Algorithm Learning Task")
parser.add_argument("--config", type=str, default="config/default.yaml", help="config file path.")
parser.add_argument("--force_restart", type=bool, default=False, help="if True start training from scratch.")

def create_experiment(config):
    


def run_experiment(args):                                                                                                         """Runs the experiment."""                                                                                                                                                                                                                                  config = create_config(args.config)                                                                                                                                                                                                                         logging.info(config.get())                                                                                                                                                                                                                                  experiment, model, data_iterator, tr, logger, device = create_experiment(config)                                                                                                                                                                            if not args.force_restart:                                                                                                        if experiment.is_resumable():                                                                                                     experiment.resume()                                                                                                   else:                                                                                                                             experiment.force_restart()                                                                                                                                                                                                                              train(experiment, model, config, data_iterator, tr, logger, device)      

if __name__ == '__main__':
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    run_experiment(args)


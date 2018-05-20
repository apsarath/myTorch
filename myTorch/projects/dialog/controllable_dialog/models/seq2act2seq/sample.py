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

from myTorch.projects.dialog.controllable_dialog.models.seq2act2seq.seq2act2seq import Seq2Act2Seq

parser = argparse.ArgumentParser(description="seq2act2seq")
parser.add_argument("--config", type=str, default="config/opus/default.yaml", help="config file path.")

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

    model = Seq2Act2Seq(config.emb_size_src, len(corpus.str_to_id), len(config.act_anotation_datasets),
                        corpus.num_acts(tag=config.act_anotation_datasets[0]),
                        config.act_emb_dim, config.act_layer_dim,
                        config.hidden_dim_src, config.hidden_dim_tgt,
                        corpus.str_to_id[config.pad], bidirectional=config.bidirectional,
                        nlayers_src=config.nlayers_src, nlayers_tgt=config.nlayers_tgt,
                        dropout_rate=config.dropout_rate).to(device)
    logging.info("Num params : {}".format(model.num_parameters))
    logging.info("Act annotation datasets : {}".format(config.act_anotation_datasets))

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

    start_time = time.time()
    num_batches = 0
    f = open("samples_{}.txt".format(config.ex_name), "w")
    count = 1
    for mini_batch in itr:
        print("Count : {}".format(count))
        count += 1
        if count > 100:
            break
        num_batches = mini_batch["num_batches"]
        model.zero_grad()
        output_logits, curr_act_logits, next_act_logits = model(
                        mini_batch["sources"].to(device), 
                        mini_batch["sources_len"].to(device),
                        mini_batch["targets_input"].to(device),
                        [mini_batch["{}_source_acts".format(act_anotation_dataset)].to(device) \
                        for act_anotation_dataset in config.act_anotation_datasets],
                        is_training=True if mode=="train" else False)


        probs = torch.nn.functional.softmax(output_logits, dim=2)
        probs, word_ids = probs.topk(20000, dim=2)
        probs = torch.nn.functional.softmax(probs, dim=2)
        probs, word_ids = probs.detach().cpu().numpy(), word_ids.detach().cpu().numpy()
        
        samples =[]
        for e_id in range(len(word_ids)):
            sample = []
            for t in range(len(word_ids[e_id])):
                word_id = np.random.choice(word_ids[e_id][t], p=probs[e_id][t])
                if word_id != data_reader.corpus.str_to_id[config.eou]:
                    sample.append(data_reader.corpus.id_to_str[word_id])
                else:
                    continue
            samples.append(sample)

        source_texts = []
        text = {}
        for key in ["sources", "targets_output"]:
            utterances = mini_batch[key].detach().cpu().numpy()
            text[key] = []
            for utterance in utterances:
                utterance_list = []
                for word_id in utterance:
                    if word_id == data_reader.corpus.str_to_id[config.eou] or \
                       word_id == data_reader.corpus.str_to_id[config.pad]:
                        continue
                    else:
                        utterance_list.append(data_reader.corpus.id_to_str[word_id])
                text[key].append(utterance_list)

        for source_text, target_text, sample in zip(text["sources"], text["targets_output"], samples):
            f.write("Source : {} \n GroundTruth : {} \n Sample : {}\n\n".format(
                " ".join(source_text),
                " ".join(target_text),
                " ".join(sample)))

    f.close()
                

def run_experiment(args):
    """Runs the experiment."""

    config = create_config(args.config)

    logging.info(config.get())

    experiment, model, data_reader, tr, logger, device = create_experiment(config)

    experiment.resume("best_model", "model")

    for mode in ["valid"]:
        tr.mini_batch_id[mode] = 0
        run_epoch(0, mode, experiment, model, config, data_reader, tr, logger, device)

        
if __name__ == '__main__':
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run_experiment(args)

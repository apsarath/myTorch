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
from myTorch.projects.dialog.controllable_dialog.data_readers.twitter_corpus import Twitter

from myTorch.projects.dialog.controllable_dialog.models.seq2act2seq.seq2act2seq import Seq2Act2Seq

parser = argparse.ArgumentParser(description="seq2act2seq")
parser.add_argument("--config", type=str, default="config/opus/default.yaml", help="config file path.")

def _safe_exp(x):
    try:
        return math.exp(x)
    except:
        return 0.0

def get_dataset(config):
    if config.dataset == "opus":
        corpus = OPUS(config)
    elif config.dataset == "twitter":
        corpus = Twitter(config)
    return corpus

def create_experiment(config):
    device = torch.device(config.device)
    logging.info("using {}".format(config.device))


    save_dir = os.path.join(config.save_dir, "_".join(config.act_anotation_datasets))
    print("Saving at {}".format(save_dir))

    experiment = GenExperiment(config.name, save_dir)
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
                        dropout_rate=config.dropout_rate, device=device).to(device)
    logging.info("Num params : {}".format(model.num_parameters))
    logging.info("Act annotation datasets : {}".format(config.act_anotation_datasets))

    experiment.register("model", model)

    optimizer = get_optimizer(model.parameters(), config)
    model.register_optimizer(optimizer)

    tr = MyContainer()

    tr.mini_batch_id, tr.loss_per_epoch, tr.rewards_avg = {}, {}, {}
    tr.epoch_id = 0

    for mode in ["train", "valid", "test"]:
        tr.mini_batch_id[mode] = 0
        tr.loss_per_epoch[mode] = []
        tr.rewards_avg[mode] = []

    experiment.register("train_statistics", tr)

    return experiment, model, reader, tr, logger, device

def run_epoch(epoch_id, mode, experiment, model, config, data_reader, tr, logger, device):
    
    itr = data_reader.itr_generator(mode, tr.mini_batch_id[mode])
    weight_mask = torch.ones(len(data_reader.corpus.str_to_id)).to(device)
    weight_mask[data_reader.corpus.str_to_id[config.pad]] = 0
    logit_loss_fn = torch.nn.CrossEntropyLoss(weight=weight_mask)
    rl_logit_loss_fn = torch.nn.CrossEntropyLoss(weight=weight_mask, reduce=False)
    act_loss_fn = torch.nn.CrossEntropyLoss()

    start_time = time.time()
    num_batches = 0
    loss_per_epoch = []
    for mini_batch in itr:
        num_batches = mini_batch["num_batches"]
        model.zero_grad()

        # forward prop
        output_logits, curr_act_logits, next_act_logits = model(
                        mini_batch["sources"].to(device), 
                        mini_batch["sources_len"].to(device),
                        mini_batch["targets_input"].to(device),
                        [mini_batch["{}_source_acts".format(act_anotation_dataset)].to(device) \
                        for act_anotation_dataset in config.act_anotation_datasets],
                        is_training=True if mode=="train" else False)

        # sample action i.e. sample next act.
        log_taken_pvals, actions, pvals = [], [], []
        for i in range(len(config.act_anotation_datasets)):
            pvals.append(torch.nn.functional.softmax(next_act_logits[i], dim=1))
            if mode == "train":
                actions.append(torch.multinomial(pvals[-1], 1).detach())
            else:
                actions.append(torch.max(pvals[-1], dim=1)[1].unsqueeze(1))

            log_taken_pvals.append(torch.log(torch.gather(pvals[-1], 1, actions[-1])).squeeze(1))

        # sample sentence
        """
        temp = 0.4
        probs = torch.nn.functional.softmax(output_logits/temp, dim=2)
        probs = probs.detach().cpu().numpy()
        word_ids = np.arange(len(data_reader.corpus.id_to_str))
            
        samples =[]
        for e_id in range(len(probs)):
            sample = []
            for t in range(len(probs[e_id])):
                word_id = np.random.choice(word_ids, p=probs[e_id][t])
                if word_id != data_reader.corpus.str_to_id[config.eou]:
                    sample.append(word_id)
                else:
                    sample.append(word_id)
                    continue
            samples.append(sample)


        samples_lens = [ len(sample) for sample in samples]
        sorted_indices = np.argsort(samples_lens)[::-1]
        samples = [samples[idx] for idx in sorted_indices]

        samples = data_reader.corpus.pad_sentences(samples, config.sentence_len_cut_off)
        samples_lens = torch.LongTensor(samples_lens)

        """
        # feed targets directly instead of samples
        samples_lens = mini_batch["targets_len"].detach().cpu().numpy()
        sorted_indices = np.argsort(samples_lens)[::-1]
        samples_lens = [samples_lens[idx] for idx in sorted_indices] 
        samples = torch.stack([mini_batch["targets_output"][idx] for idx in sorted_indices])
        samples_lens = torch.LongTensor(samples_lens)
        

        # compute reward for each generic sentence
        rl_loss = []
        generic_response_input, generic_response_output = data_reader.corpus.padded_generic_responses()
        avg_reward = []
        for i in range(generic_response_input.shape[0]):

            output_logits_r, _, _ = model(
                        samples.to(device),
                        samples_lens.to(device),
                        generic_response_input[i].unsqueeze(0).expand(samples.shape[0],-1).to(device),
                        [act.squeeze().to(device) for act in actions],
                        False)

            tgt_logits = generic_response_output[i].unsqueeze(0).expand(samples.shape[0],-1).to(device).contiguous().view(-1) 
            reward_per_generic_r = rl_logit_loss_fn(
                output_logits_r.contiguous().view(-1, output_logits_r.size(2)),
                tgt_logits)
            reward_per_generic_r = torch.mean(reward_per_generic_r.view(samples.shape[0],-1), dim=1).detach()
            avg_reward.append(torch.mean(reward_per_generic_r))
            # subtract from baseline
            if len(tr.rewards_avg[mode]) > 0:
                avg_so_far = torch.mean(torch.stack(tr.rewards_avg[mode]))
                reward_per_generic_r = (reward_per_generic_r - avg_so_far)

            for i in range(len(log_taken_pvals)):
                rl_loss.append(torch.mean(reward_per_generic_r * log_taken_pvals[i]))

            
        tr.rewards_avg[mode].append(torch.mean(torch.stack(avg_reward)))
        rl_loss = -torch.mean(torch.stack(rl_loss))
        loss_per_epoch.append(rl_loss.item())

        # SL loss       
        logit_loss = logit_loss_fn( output_logits.contiguous().view(-1, output_logits.size(2)), 
                mini_batch["targets_output"].to(device).contiguous().view(-1))

        curr_act_loss, next_act_loss = 0.0, 0.0
        for i, act_anotation_dataset in enumerate(config.act_anotation_datasets):
            curr_act_loss += act_loss_fn(
                            curr_act_logits[i], 
                            mini_batch["{}_source_acts".format(act_anotation_dataset)].to(device))

            next_act_loss += act_loss_fn(
                            next_act_logits[i],
                            mini_batch["{}_target_acts".format(act_anotation_dataset)].to(device))


        loss = logit_loss + config.curr_act_loss_coeff * curr_act_loss + config.next_act_loss_coeff * next_act_loss
        loss += rl_loss
        
        
        loss_per_epoch.append(logit_loss.item())
        
        running_average = np.mean(np.array(loss_per_epoch))

        if mode == "train":
            model.optimizer.zero_grad()
            loss.backward(retain_graph=False)
            torch.nn.utils.clip_grad_norm_(model.rl_parameters(), config.grad_clip_norm)
            model.optimizer.step()

        tr.mini_batch_id[mode] += 1

        if 1:#tr.mini_batch_id[mode] % 100 == 0 and mode == "train":
            logging.info("Epoch : {}, {} %: {}, time : {}".format(epoch_id, mode, (100.0*tr.mini_batch_id[mode]/num_batches), time.time()-start_time))
            logging.info("Running loss: {}, Reward : {}".format(loss_per_epoch[-1], tr.rewards_avg[mode][-1].item()))

    # Epoch level logging
    avg_loss = np.mean(np.array(loss_per_epoch))
    tr.loss_per_epoch[mode].append(avg_loss)
        
    logging.info("{}: loss: , time : {}".format(mode, avg_loss, _safe_exp(avg_loss), time.time() - start_time))


def run_experiment(args):
    """Runs the experiment."""

    config = create_config(args.config)

    logging.info(config.get())

    experiment, model, data_reader, tr, logger, device = create_experiment(config)

    experiment.resume("best_model","model")

    optimizer = get_optimizer(model.rl_parameters(), config)
    model.register_optimizer(optimizer)


    for i in range(tr.epoch_id, config.num_epochs):
        logging.info("#################### \n Epoch id : {} \n".format(i))
        for mode in ["train", "valid"]:
            tr.mini_batch_id[mode] = 0
            tr.epoch_id = i
            run_epoch(i, mode, experiment, model, config, data_reader, tr, logger, device)
        experiment.save("rl_model", "model")

    best_valid_loss = np.min(np.array(tr.loss_per_epoch["valid"]))
    logging.info("#################### \n Best Valid loss : {}, perplexity : {} \n".format(best_valid_loss,
        _safe_exp(best_valid_loss)))
        
if __name__ == '__main__':
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run_experiment(args)

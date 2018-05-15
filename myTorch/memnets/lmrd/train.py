import os
import math
import time

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import argparse
import logging

from myTorch import Experiment
from myTorch.utils.logging import Logger
from myTorch.utils import MyContainer, get_optimizer, create_config
from myTorch.memnets.lmrd.data import lmrd_data
from myTorch.memnets.lmrd.recurrent_with_embedding import RecurrentModel


parser = argparse.ArgumentParser(description="LMRD Sentiment Analysis Task")
parser.add_argument("--config", type=str, default="myTorch/memnets/lmrd/default.yaml", help="config file path.")
parser.add_argument("--force_restart", type=bool, default=False, help="if True start training from scratch.")


def _safe_exp(x):
    try:
        return math.exp(x)
    except:
        return 0.0


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
    
    curr_epoch_loss = []
    start_time = time.time()
    for step, (x, m, y) in enumerate(batched_data[mode]()):
        
        # Prepare the batch for pytorch.
        x = Variable(torch.from_numpy(x)).to(device)
        y = Variable(torch.from_numpy(y)).to(device)
        
        if config.inter_saving is not None:
            if tr.updates_done[mode] in config.inter_saving and mode == "train":
                experiment.save(str(tr.updates_done[mode]))
        
        model.repackage_hidden()
        
        output_logits = model(x)
        this_batch_size = x.shape[1]
        seqloss = 0
        for i in range(this_batch_size):
            final_idx = np.sum(m[:,i])-1
            seqloss += F.cross_entropy(output_logits[final_idx][i:i+1], y[i:i+1])
        seqloss /= float(this_batch_size)
        
        tr.average_loss[mode].append(seqloss.item())
        curr_epoch_loss.append(seqloss.item())

        running_average = sum(tr.average_loss[mode]) / len(tr.average_loss[mode])

        if config.use_tflogger and mode == "train":
            logger.log_scalar("running_avg_loss", running_average, tr.updates_done[mode] + 1)
            logger.log_scalar("loss", tr.average_loss[mode][-1], tr.updates_done[mode] + 1)
            logger.log_scalar("running_perplexity", _safe_exp(running_average), tr.updates_done[mode] + 1)
            logger.log_scalar("inst_perplexity", _safe_exp(tr.average_loss[mode][-1]), tr.updates_done[mode] + 1)

        if mode == "train":
            model.optimizer.zero_grad()
            seqloss.backward(retain_graph=False)
            torch.nn.utils.clip_grad_norm(model.parameters(), config.grad_clip_norm)
            model.optimizer.step()

        tr.updates_done[mode] +=1
        step += 1
        if tr.updates_done[mode] % 1e6 == 0 and mode == "train":
            logging.info("Epoch : {}, {} %: {}, step : {}".format(epoch_id, mode, (100.0*step/batch_data[mode].num_batches), tr.updates_done[mode]))
            logging.info("inst loss: {}, inst perp: {}".format(tr.average_loss[mode][-1], _safe_exp(tr.average_loss[mode][-1])))
            
    curr_epoch_avg_loss = np.mean(np.array(curr_epoch_loss))
    tr.average_loss_per_epoch[mode].append(curr_epoch_avg_loss)

    logging.info("Avg {} loss: {}, BPC : {}, Avg perp: {}, time : {}".format(mode, curr_epoch_avg_loss, curr_epoch_avg_loss/0.693, _safe_exp(curr_epoch_avg_loss), time.time() - start_time))

    if mode != "train":
        logger.log_scalar("loss_{}".format(mode), curr_epoch_avg_loss, epoch_id+1)
        logger.log_scalar("perplexity_{}".format(mode), _safe_exp(curr_epoch_avg_loss), epoch_id+1)


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

    batch_data = {}
    # Max number of words:
    # Train: 2470
    # Valid: 1723
    # Test:  2278
    lmrd_data_kwargs = {'source_dict': os.path.join(config.data, "imdb.vocab"),
                        'batch_size': config.batch_size,
                        'n_words_source': config.n_words_source,
                        'char_level': False}
    batch_data['train'] = lmrd_data(source=os.path.join(config.data, 'train'),
                                    maxlen=config.maxlen,
                                    random=True,
                                    rng=np.random.RandomState(config.rseed),
                                    **lmrd_data_kwargs)
    batch_data['valid'] = lmrd_data(source=os.path.join(config.data, 'valid'),
                                    maxlen=1723,
                                    random=False,
                                    **lmrd_data_kwargs)
    batch_data['valid'] = lmrd_data(source=os.path.join(config.data, 'valid'),
                                    maxlen=2278,
                                    random=False,
                                    **lmrd_data_kwargs)
    
    vocab_len = config.n_words_source   # Assuming this is not `-1` since there are too many possible words in the vocab (89527).
    
    model = RecurrentModel(device, 2, vocab_len, config.input_emb_size,
                           num_layers=config.num_layers, layer_size=config.layer_size,
                           cell_name=config.model, activation=config.activation,
                           output_activation="linear", layer_norm=config.layer_norm,
                           identity_init=config.identity_init, chrono_init=config.chrono_init,
                           t_max=config.bptt, memory_size=config.memory_size, k=config.k, use_relu=config.use_relu).to(device)
    experiment.register_model(model)

    optimizer = get_optimizer(model.parameters(), config)
    model.register_optimizer(optimizer)

    tr = MyContainer()

    tr.mini_batch_id, tr.updates_done, tr.average_loss, tr.average_loss_per_epoch = {}, {}, {}, {}

    for mode in ["train", "valid", "test"]:
        tr.mini_batch_id[mode] = 0
        tr.updates_done[mode] = 0
        tr.average_loss[mode] = []
        tr.average_loss_per_epoch[mode] = []
        

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

    # Train, validate
    for i in range(config.num_epochs):
        logging.info("\n#####################\n Epoch id: {}\n".format(i+1))
        for mode in ["train", "valid"]:
            tr.mini_batch_id[mode] = 0
            run_epoch(i, mode, experiment, model, config, batched_data, tr, logger, device)
    
    # Test
    run_epoch(0, "test", experiment, model, config, batched_data, tr, logger, device)

    logging.info("\n#####################\n Best Model\n")
    min_id = np.argmin(np.array(tr.average_loss_per_epoch["valid"]))
    valid_loss = tr.average_loss_per_epoch["valid"][min_id] / 0.693
    logging.info("Best Valid BPC : {}, perplexity : {}".format(valid_loss, _safe_exp(valid_loss)))
    test_loss = tr.average_loss_per_epoch["test"][min_id] / 0.693
    logging.info("Best Test BPC : {}, perplexity : {}".format(test_loss, _safe_exp(test_loss)))


if __name__ == '__main__':
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    run_experiment(args)

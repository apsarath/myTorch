import argparse
import logging
import gin
import torch
from torch.autograd import Variable
import torch.optim as optim
from myTorch.utils import MyContainer, create_config, get_optimizer
from myTorch import Logger
from myTorch import Experiment
from MLP import MLP




from myTorch.task.mnist import MNISTData

from MLP import *

parser = argparse.ArgumentParser(description="MNIST Classification Task")
parser.add_argument("--config", type=str, default="config/default.yaml", help="config file path.")
parser.add_argument("--force_restart", type=bool, default=False, help="if True start training from scratch.")
parser.add_argument("--device", type=str, default="cuda")


def compute_accuracy(model, data_iterator, data_tag, device):

    accuracy = 0.0
    total = 0.0

    while True:
        data = data_iterator.next(data_tag)
        if data is None:
            break
        x = torch.from_numpy(data['x']).to(device)
        y = torch.from_numpy(data['y']).to(device)
        output = model(x)
        _, pred = torch.max(output, 1)
        _, target = torch.max(y, 1)
        accuracy += (pred==target).sum().item()
        total += len(pred)

    accuracy /= total
    return accuracy

def train(experiment, model, config, data_iterator, tr, logger, device):
    """Training loop.
    Args:
        experiment: experiment object.
        model: model object.
        config: config dictionary.
        data_iterator: data iterator object
        tr: training statistics dictionary.
        logger: logger object.
    """

    for i in range(tr.epochs_done, config.num_epochs):

        data_iterator.reset_iterator()
        avg_loss = 0
        while True:
            data = data_iterator.next("train")

            if data is None:
                break

            x = torch.from_numpy(data['x']).to(device)
            y = torch.from_numpy(data['y']).to(device)
            model.optimizer.zero_grad()
            output = model(x)
            loss = F.binary_cross_entropy_with_logits(output, y)
            avg_loss += loss
            loss.backward()
            model.optimizer.step()
        avg_loss /= data_iterator._state.batches["train"]
        tr.train_loss.append(avg_loss)
        logger.log_scalar("training loss per epoch", avg_loss, i + 1)

        val_acc = compute_accuracy(model, data_iterator, "valid", device)
        test_acc = compute_accuracy(model, data_iterator, "test", device)
        tr.valid_acc.append(val_acc)
        tr.test_acc.append(test_acc)
        logger.log_scalar("valid acc per epoch", val_acc, i + 1)
        logger.log_scalar("test acc per epoch", test_acc, i + 1)

        experiment.save()

        tr.epochs_done += 1


def create_experiment(config):


    device = torch.device(config.device)
    logging.info("using {}".format(config.device))

    experiment = Experiment(config.name, config.save_dir)

    logger = None
    if config.use_tflogger:
        logger = Logger(config.tflog_dir)

    torch.manual_seed(config.rseed)


    model = MLP(num_hidden_layers=config.num_hidden_layers, hidden_layer_size=config.hidden_layer_size,
        activation=config.activation, input_dim=config.input_dim, output_dim=config.output_dim).to(device)

    data_iterator = MNISTData(batch_size=config.batch_size, seed=config.data_iterator_seed, use_one_hot=config.use_one_hot)

    optimizer = get_optimizer(model.parameters(), config)
    model.register_optimizer(optimizer)


    training_statistics = MyContainer()
    training_statistics.train_loss = []
    training_statistics.valid_acc = []
    training_statistics.test_acc = []
    training_statistics.epochs_done = 0

    experiment.register_experiment(model, config, logger, training_statistics, data_iterator)

    return experiment, model, data_iterator, training_statistics, logger, device


def run_experiment(args):
    """Runs the experiment."""

    config = create_config(args.config)
    config.device = args.device

    logging.info(config.get())

    experiment, model, data_iterator, training_statistics, logger, device = create_experiment(config)

    if not args.force_restart:
        if experiment.is_resumable():
            experiment.resume()
    else:
        experiment.force_restart()

    train(experiment, model, config, data_iterator, training_statistics, logger, device)



if __name__ == '__main__':
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO)
  run_experiment(args)





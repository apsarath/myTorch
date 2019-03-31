from absl import flags
from absl import app
import logging
import gin
import torch
from torch.autograd import Variable
import torch.optim as optim
from myTorch.utils import MyContainer, load_gin_configs
from myTorch import Logger
from myTorch import Experiment




from myTorch.task.mnist import MNISTData

from MLP import *

flags.DEFINE_multi_string("gin_files", [], "List of paths to gin configuration files.")
flags.DEFINE_multi_string("gin_bindings", [], "Gin bindings to override the values set in the config files "
                                              "(e.g. 'DQNAgent.epsilon_train=0.1',"
                                              "'create_environment.game_name='Pong'').")

FLAGS = flags.FLAGS


@gin.configurable
def train(batch_size=10, num_epochs=10):


    print(batch_size)
    print(num_epochs)

    return








@gin.configurable
def create_experiment(experiment_name, save_dir, use_tflogger=False, tflog_dir=None, random_seed=5):

    torch.manual_seed(random_seed)

    experiment = Experiment(experiment_name, save_dir)

    data_gen = MNISTData()

    model = MLP()
    optimizer = optim.Adam(model.parameters(), lr=0.01, betas=[0.9, 0.999])
    model.register_optimizer(optimizer)

    training_statistics = MyContainer()
    training_statistics.train_loss = []
    training_statistics.valid_acc = []
    training_statistics.test_acc = []
    training_statistics.epochs_done = 0

    logger = None
    if use_tflogger:
        logger = Logger(tflog_dir)




def main(unused_argv):
    """Main method.

    Args:
        unused_argv: Arguments (unused).
    """

    logging.basicConfig(level=logging.INFO)
    load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
    train()



if __name__ == '__main__':
  app.run(main)





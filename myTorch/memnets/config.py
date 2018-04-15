import os
from myTorch.utils import *


def copy_task_RNN():

    config = MyContainer()

    config.name = "ex1"

    # model specific details
    config.model = "GRU"
    config.input_size = 9
    config.output_size = 9
    config.num_layers = 1
    config.layer_size = [256]
    config.activation = "tanh"

    # optimization specific details
    config.optim_name = "Adam"  # valid optimizer names : Adadelta, Adagrad, Adam, RMSprop, SGD
    config.lr = 1e-3
    config.rho = 0.9
    config.eps = 1e-8
    config.weight_decay = 0
    config.lr_decay = 0
    config.beta_0 = 0.9
    config.beta_1 = 0.999
    config.alpha = 0.99
    config.momentum = 0
    config.centered = False
    config.dampening = 0
    config.nesterov = False
    config.grad_clip = [-10, 10]

    config.max_steps = 1000000
    config.rseed = 5
    config.use_gpu = True


    # task specific details
    config.num_bits = 8
    config.min_len = 1
    config.max_len = 20

    # saving details
    config.use_tflogger = True
    config.tflog_dir = os.environ["LOGDIR"]+"/copyRNN/{}/{}/".format(config.model, config.name)
    config.save_dir = os.environ["SAVEDIR"]+"/copyRNN/{}/{}/".format(config.model, config.name)
    config.save_every_n = 100
    config.force_restart = True
    create_folder(config.tflog_dir)
    create_folder(config.save_dir)

    return config

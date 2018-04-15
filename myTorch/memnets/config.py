import os
from myTorch.utils import *


def copy_task_RNN():

    config = MyContainer()

    config.name = "ex1"

    # model specific details
    config.model = "LSTM"
    config.input_size = 8
    config.output_size = 8
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

    config.max_steps = 100000000
    config.rseed = 5
    config.use_gpu = True


    # task specific details
    config.task = "copy"
    config.num_bits = 8
    config.min_len = 1
    config.max_len = 20
    config.batch_size = 10

    # saving details
    config.use_tflogger = True
    config.tflog_dir = os.environ["LOGDIR"]+"/copyRNN/{}/{}/".format(config.model, config.name)
    config.save_dir = os.environ["SAVEDIR"]+"/copyRNN/{}/{}/".format(config.model, config.name)
    config.save_every_n = 10000
    config.force_restart = True
    create_folder(config.tflog_dir)
    create_folder(config.save_dir)

    return config

def repeat_copy_task_RNN():

    config = MyContainer()

    config.name = "ex1"

    # model specific details
    config.model = "LSTM"
    config.input_size = 8
    config.output_size = 8
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

    config.max_steps = 100000000
    config.rseed = 5
    config.use_gpu = True


    # task specific details
    config.task = "repeat_copy"
    config.num_bits = 8
    config.min_len = 1
    config.max_len = 10
    config.min_repeat = 1
    config.max_repeat = 10
    config.batch_size = 10

    # saving details
    config.use_tflogger = True
    config.tflog_dir = os.environ["LOGDIR"]+"/repeat_copyRNN/{}/{}/".format(config.model, config.name)
    config.save_dir = os.environ["SAVEDIR"]+"/repeat_copyRNN/{}/{}/".format(config.model, config.name)
    config.save_every_n = 10000
    config.force_restart = True
    create_folder(config.tflog_dir)
    create_folder(config.save_dir)

    return config

def associative_recall_task_RNN():

    config = MyContainer()

    config.name = "ex1"

    # model specific details
    config.model = "LSTM"
    config.input_size = 8
    config.output_size = 8
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

    config.max_steps = 100000000
    config.rseed = 5
    config.use_gpu = True


    # task specific details
    config.task = "associative_recall"
    config.num_bits = 8
    config.min_len = 2
    config.max_len = 6
    config.block_len = 3
    config.batch_size = 10

    # saving details
    config.use_tflogger = True
    config.tflog_dir = os.environ["LOGDIR"]+"/associative_recallRNN/{}/{}/".format(config.model, config.name)
    config.save_dir = os.environ["SAVEDIR"]+"/associative_recallRNN/{}/{}/".format(config.model, config.name)
    config.save_every_n = 10000
    config.force_restart = True
    create_folder(config.tflog_dir)
    create_folder(config.save_dir)

    return config
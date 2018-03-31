import myTorch
from myTorch.utils import *
import torch.optim as optim


def copy_task_RNN():

    config = MyContainer()

    config.name = "copy_task"

    # model specific details
    config.model = "RNN"
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
    config.tflog_dir = "copyRNN/{}/tflog/p2/".format(config.model)
    config.save_dir = "copyRNN/{}/p2/".format(config.model)
    create_folder(config.tflog_dir)
    create_folder(config.save_dir)

    return config

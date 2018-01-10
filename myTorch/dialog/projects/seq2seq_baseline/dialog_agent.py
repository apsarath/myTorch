import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


import myTorch
from myTorch.utils import my_variable
from myTorch.dialog.projects.seq2seq_baseline.networks import *


class DialogAgent(object):
    def __init__(self, encoder, decoder):
        self._encoder = encoder
        self._decoder = decoder

    def 
        
        

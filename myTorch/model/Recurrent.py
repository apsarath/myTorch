import torch
import torch.nn as nn
import numpy as np

import myTorch
from myTorch.memory import RNNCell, GRUCell, LSTMCell
from myTorch.utils import act_name

from torch.autograd import Variable
import torch.nn.functional as F

class Recurrent(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, mname="LSTM", activation="tanh", output_activation=None):
        
        super(Recurrent, self).__init__()

        
        if mname=="LSTM":
            self.Cell = LSTMCell(input_size, hidden_size)
        elif mname=="GRU":
            self.Cell = GRUCell(input_size, hidden_size)
        elif mname=="RNN":
            self.Cell = RNNCell(input_size, hidden_size, activation=activation)

        self.h_prev = self.Cell.reset_hidden()

        self.output_size = output_size
        if output_activation == None:
            self.output_activation = None
        elif output_activation == "sigmoid":
            self.output_activation = torch.sigmoid
        elif output_activation == "LogSoftmax":
            self.output_activation = torch.nn.LogSoftmax


        self.W_h2o = nn.Parameter(torch.Tensor(hidden_size, output_size))
        self.b_o = nn.Parameter(torch.Tensor(output_size))

        self.reset_parameters()

        
    def forward(self, input):
        
        h = self.Cell(input, self.h_prev)
        output = torch.add(torch.mm(h["h"], self.W_h2o), self.b_o)
        if self.output_activation is not None:
            output = self.output_activation(output)
        self.h_prev = h
        return output

    def reset_hidden(self):
        self.h_prev = self.Cell.reset_hidden()

    def reset_parameters(self):

        nn.init.xavier_normal(self.W_h2o, gain=nn.init.calculate_gain(act_name(self.output_activation)))
        nn.init.constant(self.b_o, 0)


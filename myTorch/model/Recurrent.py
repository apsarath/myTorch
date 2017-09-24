import torch
import torch.nn as nn
import numpy as np

import myTorch
from myTorch.memory import RNNCell, GRUCell, LSTMCell
from myTorch.utils import act_name

from torch.autograd import Variable
import torch.nn.functional as F

class Recurrent(nn.Module):

    def __init__(self, input_size, output_size, num_layers=1, layer_size=[10], mname="LSTM", activation="tanh", output_activation=None, use_gpu=False):
        
        super(Recurrent, self).__init__()

        self.use_gpu = use_gpu

        self.Cells = []
        if mname=="LSTM":
            cell = LSTMCell
        elif mname=="GRU":
            cell = GRUCell
        elif mname=="RNN":
            cell = RNNCell

        self.num_layers = num_layers
        self.Cells.append(cell(input_size, layer_size[0], activation=activation, use_gpu=use_gpu))
        for i in range(1, num_layers):
            self.Cells.append(cell(layer_size[i-1], layer_size[i], activation=activation, use_gpu=use_gpu))

        self.reset_hidden()

        self.list_of_modules = nn.ModuleList(self.Cells)
        self.output_size = output_size
        if output_activation == None:
            self.output_activation = None
        elif output_activation == "sigmoid":
            self.output_activation = torch.sigmoid
        elif output_activation == "LogSoftmax":
            self.output_activation = torch.nn.LogSoftmax


        self.W_h2o = nn.Parameter(torch.Tensor(layer_size[-1], output_size))
        self.b_o = nn.Parameter(torch.Tensor(output_size))

        self.reset_parameters()

        
    def forward(self, input):
        
        h = []
        h.append(self.Cells[0](input, self.h_prev[0]))
        for i, cell in enumerate(self.Cells):
            if i!=0:
                h.append(cell(h[i-1]["h"], self.h_prev[i]))
        output = torch.add(torch.mm(h[-1]["h"], self.W_h2o), self.b_o)
        if self.output_activation is not None:
            output = self.output_activation(output)
        self.h_prev = h
        return output

    def reset_hidden(self):
        self.h_prev = []
        for cell in self.Cells:
            self.h_prev.append(cell.reset_hidden())

    def reset_parameters(self):

        nn.init.xavier_normal(self.W_h2o, gain=nn.init.calculate_gain(act_name(self.output_activation)))
        nn.init.constant(self.b_o, 0)



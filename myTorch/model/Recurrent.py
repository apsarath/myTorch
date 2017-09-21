import torch
import torch.nn as nn
import numpy as np

import myTorch
from myTorch.memory import RNNCell

from torch.autograd import Variable

class Recurrent(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, activation="tanh", output_activation=None):
        
        super(Recurrent, self).__init__()

        
        self.Cell = RNNCell(input_size, hidden_size, activation)

        self.hidden_size = hidden_size

        self.output_size = output_size
        if output_activation == None:
            self.output_activation = None
        elif output_activation == "sigmoid":
            self.output_activation = torch.sigmoid
        elif output_activation == "LogSoftmax":
            self.output_activation = torch.LogSoftmax

        self.h_prev = Variable(torch.Tensor(np.zeros((1,hidden_size))))

        self.W_h2o = nn.Parameter(torch.Tensor(hidden_size, output_size))
        self.b_o = nn.Parameter(torch.Tensor(output_size))

        self.reset_parameters()

        
    def forward(self, input):
        
        h = self.Cell(input, self.h_prev)
        output = torch.add(torch.mm(h, self.W_h2o), self.b_o)
        if self.output_activation is not None:
            output = self.output_activation(output)
        self.h_prev = h
        return output

    def reset_hidden(self):
        self.h_prev = Variable(torch.Tensor(np.zeros((1,self.hidden_size))))

    def reset_parameters(self):

        for weight in self.parameters():
            weight.data.uniform_()

    def num_parameters(self):
        return sum([np.prod(p.size()) for p in self.parameters()])
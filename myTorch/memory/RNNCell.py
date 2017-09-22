import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable

import myTorch
from myTorch.utils import act_name


class RNNCell(nn.Module):

    def __init__(self, input_size, hidden_size, activation="tanh"):
        
        super(RNNCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "sigmoid":
            self.activation = torch.sigmoid 
        
        self.W_i2h = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_h2h = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_h = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

        
    def forward(self, input, last_hidden):
        
        c_input = torch.cat((input, last_hidden["h"]), 1)
        W_h = torch.cat((self.W_i2h, self.W_h2h), 0)
        pre_hidden = torch.add(torch.mm(c_input, W_h), self.b_h)
        h = self.activation(pre_hidden)

        hidden = {}
        hidden["h"] = h
        return hidden

    def reset_hidden(self):
        hidden = {}
        hidden["h"] = Variable(torch.Tensor(np.zeros((1,self.hidden_size))))
        return hidden

    def reset_parameters(self):

        for weight in self.parameters():
            weight.data.uniform_(-0.1,0.1)

        nn.init.xavier_normal(self.W_i2h, gain=nn.init.calculate_gain(act_name(self.activation)))
        nn.init.orthogonal(self.W_h2h)
        nn.init.constant(self.b_h, 0)

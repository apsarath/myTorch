import torch
import torch.nn as nn
import numpy as np

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
        
        c_input = torch.cat((input, last_hidden), 1)
        W_h = torch.cat((self.W_i2h, self.W_h2h), 0)
        pre_hidden = torch.add(torch.mm(c_input, W_h), self.b_h)
        hidden = self.activation(pre_hidden)
        return hidden

    def reset_parameters(self):

        for weight in self.parameters():
            weight.data.uniform_()

    def num_parameters(self):
        return sum([np.prod(p.size()) for p in self.parameters()])
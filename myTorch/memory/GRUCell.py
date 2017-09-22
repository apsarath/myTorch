import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable


class GRUCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        
        super(GRUCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.W_i2r = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_h2r = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_r = nn.Parameter(torch.Tensor(hidden_size))
        
        self.W_i2z = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_h2z = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_z = nn.Parameter(torch.Tensor(hidden_size))
        
        self.W_i2h = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_h2h = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_h = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

        self.W_r = torch.cat((self.W_i2r, self.W_h2r), 0)
        self.W_z = torch.cat((self.W_i2z, self.W_h2z), 0)

        
    def forward(self, input, last_hidden):
        
        c_input = torch.cat((input, last_hidden["h"]), 1)
        r = torch.sigmoid(torch.add(torch.mm(c_input, self.W_r), self.b_r))
        z = torch.sigmoid(torch.add(torch.mm(c_input, self.W_z), self.b_z))
        hp = torch.tanh(torch.mm(input, self.W_i2h) + (torch.mm(last_hidden["h"], self.W_h2h) * r) + self.b_h)
        h = ((1 - z) * hp) + (z * last_hidden["h"])
        
        hidden = {}
        hidden["h"] = h
        return hidden


    def reset_hidden(self):
        hidden = {}
        hidden["h"] = Variable(torch.Tensor(np.zeros((1,self.hidden_size))))
        return hidden

    def reset_parameters(self):

        nn.init.xavier_normal(self.W_i2r)
        nn.init.xavier_normal(self.W_i2z)
        nn.init.xavier_normal(self.W_i2h)
        
        nn.init.orthogonal(self.W_h2r)
        nn.init.orthogonal(self.W_h2z)
        nn.init.orthogonal(self.W_h2h)
        
        nn.init.constant(self.b_r, 0)
        nn.init.constant(self.b_z, 0)
        nn.init.constant(self.b_h, 0)
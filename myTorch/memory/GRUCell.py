import torch
import torch.nn as nn
import numpy as np

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
        
        c_input = torch.cat((input, last_hidden), 1)
        r = torch.sigmoid(torch.add(torch.mm(c_input, self.W_r), self.b_r))
        z = torch.sigmoid(torch.add(torch.mm(c_input, self.W_z), self.b_z))
        hp = torch.tanh(torch.mm(input, self.W_i2h) + (torch.mm(last_hidden, self.W_h2h) * r) + self.b_h)
        hidden = ((1 - z) * hp) + (z * last_hidden)
        return hidden

    def reset_parameters(self):

        for weight in self.parameters():
            weight.data.uniform_()

    def num_parameters(self):
        return sum([np.prod(p.size()) for p in self.parameters()])
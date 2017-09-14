import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):

    def __init__(self, data_size, hidden_size, output_size):
        
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        input_size = data_size + hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, data, last_hidden):
        input = torch.cat((data, last_hidden), 1)
        hidden = self.i2h(input)
        output = self.h2o(hidden)
        return hidden, output
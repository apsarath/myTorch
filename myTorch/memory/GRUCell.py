"""Implementation of a GRU Cell."""
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


class GRUCell(nn.Module):
    """Implementation of a GRU cell based on https://arxiv.org/pdf/1412.3555.pdf"""

    def __init__(self, input_size, hidden_size, use_gpu=False):
        """Initializes a GRU Cell.

        Args:
            input_size: int, size of the input vector.
            hidden_size: int, RNN hidden layer dimension.
            use_gpu: bool, true if using GPU.
        """
        
        super(GRUCell, self).__init__()

        self._input_size = input_size
        self._hidden_size = hidden_size
        self._use_gpu = use_gpu

        self._W_i2r = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self._W_h2r = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self._b_r = nn.Parameter(torch.Tensor(hidden_size))
        
        self._W_i2z = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self._W_h2z = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self._b_z = nn.Parameter(torch.Tensor(hidden_size))
        
        self._W_i2h = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self._W_h2h = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self._b_h = nn.Parameter(torch.Tensor(hidden_size))

        self._reset_parameters()

    def forward(self, input, last_hidden):
        """Implements forward computation of an RNN Cell.

        Args:
            input: current input vector.
            last_hidden: previous hidden state dictionary.

        Returns:
            current hidden state as a dictionary.
        """
        
        self._W_r = torch.cat((self._W_i2r, self._W_h2r), 0)
        self._W_z = torch.cat((self._W_i2z, self._W_h2z), 0)
        c_input = torch.cat((input, last_hidden["h"]), 1)

        r = torch.sigmoid(torch.add(torch.mm(c_input, self._W_r), self._b_r))
        z = torch.sigmoid(torch.add(torch.mm(c_input, self._W_z), self._b_z))
        hp = torch.tanh(torch.mm(input, self._W_i2h) + torch.mm(last_hidden["h"] * r, self._W_h2h) + self._b_h)
        h = ((1 - z) * hp) + (z * last_hidden["h"])
        
        hidden = {}
        hidden["h"] = h
        return hidden


    def reset_hidden(self):
        """Resets the hidden state for truncating the dependency."""

        hidden = {}
        hidden["h"] = Variable(torch.Tensor(np.zeros((1, self._hidden_size))))
        if self._use_gpu:
            hidden["h"] = hidden["h"].cuda()
        return hidden

    def _reset_parameters(self):
        """Initializes the GRU Cell parameters."""

        nn.init.xavier_normal(self._W_i2r)
        nn.init.xavier_normal(self._W_i2z)
        nn.init.xavier_normal(self._W_i2h)
        
        nn.init.orthogonal(self._W_h2r)
        nn.init.orthogonal(self._W_h2z)
        nn.init.orthogonal(self._W_h2h)
        
        nn.init.constant(self._b_r, 0)
        nn.init.constant(self._b_z, 0)
        nn.init.constant(self._b_h, 0)

"""Implementation of an RNN Cell."""
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class RNNCell(nn.Module):
    """Implementation of an RNN Cell."""

    def __init__(self, input_size, hidden_size, activation="tanh", use_gpu=False):
        """Initializes an RNN Cell.

        Args:
            input_size: int, size of the input vector.
            hidden_size: int, RNN hidden layer dimension.
            activation: str, hidden layer activation function.
            use_gpu: bool, true if using GPU.
        """
        
        super(RNNCell, self).__init__()

        self._input_size = input_size
        self._hidden_size = hidden_size
        self._activation = activation
        self._use_gpu = use_gpu

        if self._activation == "tanh":
            self._activation_fn = F.tanh
        elif self._activation == "sigmoid":
            self._activation_fn = F.sigmoid
        elif self._activation == "relu":
            self._activation_fn = F.relu
        
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

        c_input = torch.cat((input, last_hidden["h"]), 1)
        W_h = torch.cat((self._W_i2h, self._W_h2h), 0)
        pre_hidden = torch.add(torch.mm(c_input, W_h), self._b_h)
        h = self._activation_fn(pre_hidden)

        hidden = {}
        hidden["h"] = h
        return hidden

    def reset_hidden(self, batch_size):
        """Resets the hidden state for truncating the dependency."""

        hidden = {}
        hidden["h"] = Variable(torch.Tensor(np.zeros((batch_size, self._hidden_size))))
        if self._use_gpu:
            hidden["h"] = hidden["h"].cuda()
        return hidden

    def _reset_parameters(self):
        """Initializes the RNN Cell parameters."""

        nn.init.xavier_normal(self._W_i2h, gain=nn.init.calculate_gain(self._activation))
        nn.init.orthogonal(self._W_h2h)
        nn.init.constant(self._b_h, 0)

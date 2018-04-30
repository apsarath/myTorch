"""Implementation of an RNN Cell."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNCell(nn.Module):
    """Implementation of an RNN Cell."""

    def __init__(self, device, input_size, hidden_size, activation="tanh", layer_norm=False, identity_init=False):
        """Initializes an RNN Cell.

        Args:
            device: torch device object.
            input_size: int, size of the input vector.
            hidden_size: int, RNN hidden layer dimension.
            activation: str, hidden layer activation function.
            layer_norm: bool, if True, applies layer normalization.
            identity_init: bool, if true, initializes hidden matrix with identity matrix.
        """
        
        super(RNNCell, self).__init__()

        self._device = device
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._activation = activation
        self._layer_norm = layer_norm
        self._identity_init = identity_init

        if self._activation == "tanh":
            self._activation_fn = F.tanh
        elif self._activation == "sigmoid":
            self._activation_fn = F.sigmoid
        elif self._activation == "relu":
            self._activation_fn = F.relu
        
        self._W_i2h = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self._W_h2h = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self._b_h = nn.Parameter(torch.Tensor(hidden_size))

        if self._layer_norm:
            self._ln = nn.LayerNorm(hidden_size)

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
        if self._layer_norm:
            pre_hidden = self._ln(pre_hidden)
        h = self._activation_fn(pre_hidden)

        hidden = {}
        hidden["h"] = h
        return hidden

    def reset_hidden(self, batch_size):
        """Resets the hidden state for truncating the dependency."""

        hidden = {}
        hidden["h"] = torch.Tensor(np.zeros((batch_size, self._hidden_size))).to(self._device)
        return hidden

    def _reset_parameters(self):
        """Initializes the RNN Cell parameters."""

        nn.init.xavier_normal(self._W_i2h, gain=nn.init.calculate_gain(self._activation))
        if self._identity_init:
            nn.init.eye_(self._W_h2h)
        else:
            nn.init.orthogonal(self._W_h2h)
        nn.init.constant(self._b_h, 0)

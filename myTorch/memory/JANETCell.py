"""Implementation of a JANET Cell."""
import torch
import torch.nn as nn
import numpy as np


class JANETCell(nn.Module):
    """Implementation of a JANET Cell based on https://arxiv.org/pdf/1804.04849.pdf"""

    def __init__(self, device, input_size, hidden_size, layer_norm=False, chrono_init=False, t_max=10):
        """Initializes a JANET Cell.

        Args:
            device: torch device object.
            input_size: int, size of the input vector.
            hidden_size: int, LSTM hidden layer dimension.
            layer_norm: bool, if True, applies layer normalization.
        """

        super(JANETCell, self).__init__()

        self._device = device
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._layer_norm = layer_norm
        self._chrono_init = chrono_init
        self._t_max = t_max

        self._W_x2f = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self._W_h2f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self._b_f = nn.Parameter(torch.Tensor(hidden_size))

        self._W_x2c = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self._W_h2c = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self._b_c = nn.Parameter(torch.Tensor(hidden_size))

        if self._layer_norm:
            self._ln = nn.LayerNorm(hidden_size)

        self._reset_parameters()

    def forward(self, input, last_hidden):
        """Implements forward computation of an LSTM Cell.

        Args:
            input: current input vector.
            last_hidden: previous hidden state dictionary.

        Returns:
            current hidden state as a dictionary.
        """

        self._W_f = torch.cat((self._W_x2f, self._W_h2f), 0)
        self._W_c = torch.cat((self._W_x2c, self._W_h2c), 0)

        c_input = torch.cat((input, last_hidden["h"]), 1)
        f = torch.sigmoid(torch.mm(c_input, self._W_f) + self._b_f)

        cp_input = torch.cat((input, last_hidden["h"]), 1)
        cp = torch.mm(cp_input, self._W_c) + self._b_c
        if self._layer_norm:
            cp = self._ln(cp)
        cp = torch.tanh(cp)

        c = f * last_hidden["h"] + (1-f) * cp

        h = c

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

        nn.init.xavier_normal_(self._W_x2f)
        nn.init.xavier_normal_(self._W_x2c)

        nn.init.orthogonal_(self._W_h2f)
        nn.init.orthogonal_(self._W_h2c)

        nn.init.constant_(self._b_f, 1)
        nn.init.constant_(self._b_c, 0)

        if self._chrono_init:
            print(self._t_max)
            b_f = torch.from_numpy(np.log(np.random.randint(1, self._t_max + 1, size=self._hidden_size)))
            self._b_f.data.copy_(b_f)

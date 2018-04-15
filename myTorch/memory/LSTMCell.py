"""Implementation of an LSTM Cell."""
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


class LSTMCell(nn.Module):
    """Implementation of an LSTM Cell based on https://arxiv.org/pdf/1308.0850.pdf"""

    def __init__(self, input_size, hidden_size, use_gpu=False):
        """Initializes an LSTM Cell.

        Args:
            input_size: int, size of the input vector.
            hidden_size: int, LSTM hidden layer dimension.
            use_gpu: bool, true if using GPU.
        """

        super(LSTMCell, self).__init__()

        self._input_size = input_size
        self._hidden_size = hidden_size
        self._use_gpu = use_gpu

        self._W_x2i = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self._W_h2i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self._W_c2i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self._b_i = nn.Parameter(torch.Tensor(hidden_size))
        
        self._W_x2f = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self._W_h2f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self._W_c2f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self._b_f = nn.Parameter(torch.Tensor(hidden_size))
        
        self._W_x2o = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self._W_h2o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self._W_c2o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self._b_o = nn.Parameter(torch.Tensor(hidden_size))
        
        self._W_x2c = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self._W_h2c = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self._b_c = nn.Parameter(torch.Tensor(hidden_size))
        
        self._reset_parameters()

    def forward(self, input, last_hidden):
        """Implements forward computation of an LSTM Cell.

        Args:
            input: current input vector.
            last_hidden: previous hidden state dictionary.

        Returns:
            current hidden state as a dictionary.
        """
 
        self._W_i = torch.cat((self._W_x2i, self._W_h2i, self._W_c2i), 0)
        self._W_f = torch.cat((self._W_x2f, self._W_h2f, self._W_c2f), 0)
        self._W_o = torch.cat((self._W_x2o, self._W_h2o, self._W_c2o), 0)
        self._W_c = torch.cat((self._W_x2c, self._W_h2c), 0)
 
        c_input = torch.cat((input, last_hidden["h"], last_hidden["c"]), 1)
        i = torch.sigmoid(torch.mm(c_input, self._W_i) + self._b_i)
        f = torch.sigmoid(torch.mm(c_input, self._W_f) + self._b_f)

        cp_input = torch.cat((input, last_hidden["h"]), 1)
        cp = torch.tanh(torch.mm(cp_input, self._W_c) + self._b_c)
        c = f * last_hidden["c"] + i * cp

        o_input = torch.cat((input, last_hidden["h"], c), 1)
        o = torch.sigmoid(torch.mm(o_input, self._W_o) + self._b_o)
        
        h = o*torch.tanh(c)
        
        hidden = {}
        hidden["h"] = h
        hidden["c"] = c 
        return hidden

    def reset_hidden(self, batch_size):
        """Resets the hidden state for truncating the dependency."""

        hidden = {}
        hidden["h"] = Variable(torch.Tensor(np.zeros((batch_size, self._hidden_size))))
        hidden["c"] = Variable(torch.Tensor(np.zeros((batch_size, self._hidden_size))))
        if self._use_gpu:
            hidden["h"] = hidden["h"].cuda()
            hidden["c"] = hidden["c"].cuda()
        return hidden

    def _reset_parameters(self):
        """Initializes the RNN Cell parameters."""

        nn.init.xavier_normal(self._W_x2i)
        nn.init.xavier_normal(self._W_x2f)
        nn.init.xavier_normal(self._W_x2o)
        nn.init.xavier_normal(self._W_x2c)
        
        nn.init.orthogonal(self._W_h2i)
        nn.init.orthogonal(self._W_h2f)
        nn.init.orthogonal(self._W_h2o)
        nn.init.orthogonal(self._W_h2c)
        
        nn.init.orthogonal(self._W_c2i)
        nn.init.orthogonal(self._W_c2f)
        nn.init.orthogonal(self._W_c2o)
        
        nn.init.constant(self._b_i, 0)
        nn.init.constant(self._b_f, 1)
        nn.init.constant(self._b_o, 0)
        nn.init.constant(self._b_c, 0)

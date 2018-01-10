import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from myTorch.utils import my_variable
from myTorch.memory import RNNCell, GRUCell, LSTMCell, TARDISCell

class EncoderRNN(nn.Module):
    
    def __init__(self, input_size=300, hidden_size=1000, nlayers=1, rnn_type="LSTM", use_gpu=True):
        super(self.__class__, self).__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._use_gpu = use_gpu
        self._rnn = LSTMCell(input_size=self._rnn_input_size, hidden_size=self._rnn_hidden_size, use_gpu=use_gpu)

    def reset_hidden(self, batch_size):
        self._hidden = {}
        self._hidden["h"] = my_variable(torch.zeros(batch_size, self._rnn_hidden_size), use_gpu=self._use_gpu)
        self._hidden["c"] = my_variable(torch.zeros(batch_size, self._rnn_hidden_size), use_gpu=self._use_gpu)
    
    def forward(self, inputs):
        if self._hidden is None:
            self.reset_hidden(obs.shape[0])

        outputs = []
        for input in inputs:
            output, hidden_next = self._rnn(input, self._hidden)
            self._hidden = hidden_next
            outputs.append(output)
        return outputs

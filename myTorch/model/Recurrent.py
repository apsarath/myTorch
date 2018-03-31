"""Implementation of a generic Recurrent Network."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from myTorch.memory import RNNCell, GRUCell, LSTMCell, TARDISCell, FlatMemoryCell


class Recurrent(nn.Module):
    """Implementation of a generic Recurrent Network."""

    def __init__(self, input_size, output_size, num_layers=1, layer_size=[10],
                 cell_name="LSTM", activation="tanh", output_activation=None, use_gpu=False):
        """Initializes a recurrent network."""
        
        super(Recurrent, self).__init__()

        self._input_size = input_size
        self._output_size = output_size
        self._num_layers = num_layers
        self._layer_size = layer_size
        self._cell_name = cell_name
        self._activation = activation
        self._output_activation = output_activation
        self._use_gpu = use_gpu

        self._Cells = []
        if self._cell_name == "LSTM":
            cell = LSTMCell
        elif self._cell_name == "GRU":
            cell = GRUCell
        elif self._cell_name == "RNN":
            cell = RNNCell
        elif self._cell_name == "TARDIS":
            cell = TARDISCell
        elif self._cell_name == "FlatMemory":
            cell = FlatMemoryCell

        self._Cells.append(cell(self._input_size, self._layer_size[0],
                               activation=self._activation, use_gpu=self._use_gpu))
        for i in range(1, num_layers):
            self._Cells.append(cell(self._layer_size[i-1], self._layer_size[i],
                                   activation=self._activation, use_gpu=self._use_gpu))

        self._list_of_modules = nn.ModuleList(self._Cells)

        if self._output_activation is None:
            self._output_activation_fn = None
        elif self._output_activation == "sigmoid":
            self._output_activation_fn = F.sigmoid
        elif self._output_activation == "LogSoftmax":
            self._output_activation_fn = F.LogSoftmax

        self._W_h2o = nn.Parameter(torch.Tensor(layer_size[-1], output_size))
        self._b_o = nn.Parameter(torch.Tensor(output_size))

        self.reset_parameters()
        self._reset_hidden()
        
    def forward(self, input):
        """Implements forward computation of the model.

        Args:
            input: current input vector.

        Returns:
            model output for current time step.
        """
        
        h = []
        h.append(self._Cells[0](input, self._h_prev[0]))
        for i, cell in enumerate(self._Cells):
            if i != 0:
                h.append(cell(h[i-1]["h"], self._h_prev[i]))
        output = torch.add(torch.mm(h[-1]["h"], self._W_h2o), self._b_o)
        if self._output_activation_fn is not None:
            output = self._output_activation_fn(output)
        self._h_prev = h
        return output

    def reset_hidden(self):
        """Resets the hidden state for truncating the dependency."""

        self._h_prev = []
        for cell in self._Cells:
            self._h_prev.append(cell.reset_hidden())

    def _reset_parameters(self):
        """Initializes the parameters."""

        nn.init.xavier_normal(self._W_h2o, gain=nn.init.calculate_gain(self._output_activation))
        nn.init.constant(self._b_o, 0)



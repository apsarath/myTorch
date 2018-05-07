"""Implementation of a generic Recurrent Network."""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from myTorch.memnets.FlatMemoryCell import FlatMemoryCell
from myTorch.memory import RNNCell, GRUCell
from myTorch.projects.overfeeding.utils.ExpandableLSTMCell import ExpandableLSTMCell
from myTorch.projects.overfeeding.utils.net2net import make_h_wider, make_weight_wider_at_input


class Recurrent(nn.Module):
    """Implementation of a generic Recurrent Network."""

    def __init__(self, device, input_size, output_size, num_layers=1, layer_size=[10],
                 cell_name="LSTM", activation="tanh", output_activation="linear"):
        """Initializes a recurrent network."""

        super(Recurrent, self).__init__()

        self._device = device
        self._input_size = input_size
        self._output_size = output_size
        self._num_layers = num_layers
        self._layer_size = layer_size
        self._cell_name = cell_name
        self._activation = activation
        self._output_activation = output_activation

        self._Cells = []

        self._add_cell(self._input_size, self._layer_size[0])
        for i in range(1, num_layers):
            self._add_cell(self._layer_size[i - 1], self._layer_size[i])

        self._list_of_modules = nn.ModuleList(self._Cells)

        if self._output_activation == "linear":
            self._output_activation_fn = None
        elif self._output_activation == "sigmoid":
            self._output_activation_fn = F.sigmoid
        elif self._output_activation == "LogSoftmax":
            self._output_activation_fn = F.LogSoftmax

        self._W_h2o = nn.Parameter(torch.Tensor(layer_size[-1], output_size))
        self._b_o = nn.Parameter(torch.Tensor(output_size))

        self._reset_parameters()
        self.print_num_parameters()

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
                h.append(cell(h[i - 1]["h"], self._h_prev[i]))
        output = torch.add(torch.mm(h[-1]["h"], self._W_h2o), self._b_o)
        if self._output_activation_fn is not None:
            output = self._output_activation_fn(output)
        self._h_prev = h
        return output

    def reset_hidden(self, batch_size):
        """Resets the hidden state for truncating the dependency."""

        self._h_prev = []
        for cell in self._Cells:
            self._h_prev.append(cell.reset_hidden(batch_size))

        return self._h_prev

    def reset_hidden_randomly(self, batch_size):
        """Resets the hidden state randomly for testing the `make_cell_wider` operation.."""

        self._h_prev = []
        for cell in self._Cells:
            self._h_prev.append(cell.reset_hidden_randomly(batch_size))
        return self._h_prev

    def set_hidden(self, h):
        self._h_prev = h

    def _reset_parameters(self):
        """Initializes the parameters."""

        nn.init.xavier_normal(self._W_h2o, gain=nn.init.calculate_gain(self._output_activation))
        nn.init.constant(self._b_o, 0)

    def register_optimizer(self, optimizer):
        """Registers an optimizer for the model.

        Args:
            optimizer: optimizer object.
        """

        self.optimizer = optimizer

    def _add_cell(self, input_size, hidden_size):
        """Adds a cell to the stack of cells.

        Args:
            input_size: int, size of the input vector.
            hidden_size: int, hidden layer dimension.
        """

        if self._cell_name == "RNN":
            self._Cells.append(RNNCell(self._device, input_size, hidden_size,
                                       activation=self._activation))
        elif self._cell_name == "LSTM":
            self._Cells.append(ExpandableLSTMCell(self._device, input_size, hidden_size))
        elif self._cell_name == "GRU":
            self._Cells.append(GRUCell(self._device, input_size, hidden_size))
        elif self._cell_name == "FlatMemory":
            self._Cells.append(FlatMemoryCell(self._device, input_size, hidden_size))

    def save(self, save_dir):
        """Saves the model and the optimizer.

        Args:
            save_dir: absolute path to saving dir.
        """

        file_name = os.path.join(save_dir, "model.p")
        torch.save(self.state_dict(), file_name)

        file_name = os.path.join(save_dir, "optim.p")
        torch.save(self.optimizer.state_dict(), file_name)

    def load(self, save_dir):
        """Loads the model and the optimizer.

        Args:
            save_dir: absolute path to loading dir.
        """

        file_name = os.path.join(save_dir, "model.p")
        self.load_state_dict(torch.load(file_name))

        file_name = os.path.join(save_dir, "optim.p")
        self.optimizer.load_state_dict(torch.load(file_name))

    def print_num_parameters(self):
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        print("Num_params : {} ".format(num_params))
        return num_params

    def make_net_wider(self, new_hidden_dim):
        """
        Method to make the recurrent net wider by growing the original hidden dim to the
        size new_hidden_dim
        Args:
             new_hidden_dim: Size of the hidden dim of the recurrent net after making it wider
        """

        initial_hidden_dim = self._Cells[0]._W_x2i.shape[1]
        indices_to_copy = np.random.randint(initial_hidden_dim, size=(new_hidden_dim - initial_hidden_dim))
        replication_factor = np.bincount(indices_to_copy)

        # Growing all the RNN cells
        self._Cells[0].make_cell_wider(new_hidden_dim=new_hidden_dim,
                                       indices_to_copy=indices_to_copy,
                                       replication_factor=replication_factor,
                                       is_first_cell=True)

        for cell in self._Cells[1:]:
            cell.make_cell_wider(new_hidden_dim=new_hidden_dim,
                                 indices_to_copy=indices_to_copy,
                                 replication_factor=replication_factor,
                                 is_first_cell=False)

        # Growing the parameters of the recurrent net
        self._layer_size = new_hidden_dim

        student_w = make_weight_wider_at_input(teacher_w=self._W_h2o.data.numpy(),
                                               indices_to_copy=indices_to_copy,
                                               replication_factor=replication_factor)
        self._W_h2o.data = torch.from_numpy(student_w)

        # Growing the hidden state vectors
        for h in self._h_prev:
            for key, param in h.items():
                student_b = make_h_wider(teacher_b=param.data.numpy(),
                                         indices_to_copy=indices_to_copy)
                h[key].data = torch.from_numpy(student_b)

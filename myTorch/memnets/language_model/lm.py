"""Implementation of a generic Recurrent Network."""
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from myTorch.memnets.FlatMemoryCell import FlatMemoryCell
from myTorch.memory import RNNCell, GRUCell, JANETCell
from myTorch.projects.overfeeding.utils.ExpandableLSTMCell import ExpandableLSTMCell
from myTorch.projects.overfeeding.utils.net2net import make_h_wider, make_weight_wider_at_input, \
    make_weight_wider_at_output, make_bias_wider


class LanguageModel(nn.Module):
    """Implementation of a generic Recurrent Network."""

    def __init__(self, device, vocab_size, input_emb_size, num_layers=1, layer_size=[10],
                 cell_name="LSTM", activation="tanh", output_activation="linear",
                 layer_norm=False, identity_init=False, chrono_init=False, t_max=10,
                 memory_size=64, k=4, use_relu=True):
        """Initializes a recurrent network."""

        super(LanguageModel, self).__init__()

        self._device = device
        self._vocab_size = vocab_size
        self._input_emb_size = input_emb_size
        self._num_layers = num_layers
        self._layer_size = layer_size
        self._cell_name = cell_name
        self._activation = activation
        self._output_activation = output_activation
        self._layer_norm = layer_norm
        self._identity_init = identity_init
        self._chrono_init = chrono_init
        self._t_max = t_max
        self._memory_size = memory_size
        self._k = k
        self._use_relu = use_relu

        self._Cells = []

        self.emb = torch.nn.Embedding(self._vocab_size, self._input_emb_size)

        self._add_cell(self._input_emb_size, self._layer_size[0])
        for i in range(1, num_layers):
            self._add_cell(self._layer_size[i - 1], self._layer_size[i])

        self._list_of_modules = nn.ModuleList(self._Cells)

        if self._output_activation == "linear":
            self._output_activation_fn = None
        elif self._output_activation == "sigmoid":
            self._output_activation_fn = F.sigmoid
        elif self._output_activation == "LogSoftmax":
            self._output_activation_fn = F.LogSoftmax

        self._W_h2o = nn.Parameter(torch.Tensor(layer_size[-1], vocab_size))
        self._b_o = nn.Parameter(torch.Tensor(vocab_size))

        self._reset_parameters()
        self.print_num_parameters()

    def forward(self, input):
        input_emb = self.emb(input)
        output = []
        for t in range(input_emb.shape[0]):
            output.append(self._forward_util(input_emb[t]))
        return output

    def _forward_util(self, input):
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

    def repackage_hidden(self):
        for cell_id in range(len(self._Cells)):
            for key in self._h_prev[cell_id]:
                self._h_prev[cell_id][key] = self._h_prev[cell_id][key].detach()

    def _reset_parameters(self):
        """Initializes the parameters."""

        nn.init.xavier_normal_(self._W_h2o, gain=nn.init.calculate_gain(self._output_activation))
        nn.init.constant_(self._b_o, 0)

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
                                       activation=self._activation, layer_norm=self._layer_norm,
                                       identity_init=self._identity_init))
        elif self._cell_name == "LSTM":
            self._Cells.append(ExpandableLSTMCell(self._device, input_size, hidden_size, layer_norm=self._layer_norm,
                                                  chrono_init=self._chrono_init, t_max=self._t_max))
        elif self._cell_name == "JANET":
            self._Cells.append(JANETCell(self._device, input_size, hidden_size, layer_norm=self._layer_norm,
                                         chrono_init=self._chrono_init, t_max=self._t_max))
        elif self._cell_name == "GRU":
            self._Cells.append(GRUCell(self._device, input_size, hidden_size))
        elif self._cell_name == "FlatMemory":
            self._Cells.append(FlatMemoryCell(self._device, input_size, hidden_size,
                                              memory_size=self._memory_size, k=self._k, use_relu=self._use_relu))

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

    def can_make_net_wider(self, expanded_layer_size):
        """
        Method to check if the recurrent net can be made wider. The net can be made wider only if
        new_hidden_dim > self._layer_size[0] (ie current hidden dim)
        For now, we check for only 1 layer. This is trivial to extend for multiple layers and
        can be done on demand :)
        :param expanded_layer_size:
        :return:
        """
        flag = False
        candidate_layers = list(size for size in expanded_layer_size if size > max(self._layer_size))
        if (candidate_layers):
            flag = True
        return flag

    def make_net_wider(self, expanded_layer_size, can_make_optimizer_wider=False, use_noise=True,
                       use_random_noise=True):
        """
        Method to make the recurrent net wider by growing the original hidden dim to the
        size new_hidden_dim
        Args:
             new_hidden_dim: Size of the hidden dim of the recurrent net after making it wider
        """

        candidate_layers = list(size for size in expanded_layer_size if size > max(self._layer_size))
        if (candidate_layers):
            new_hidden_dim = candidate_layers[0]
            new_layer_size = [new_hidden_dim] * len(self._layer_size)
            logging.info("Making RNN wider. Previous width = {}, new width = {}".format(
                "_".join([str(x) for x in self._layer_size]),
                "_".join([str(x) for x in new_layer_size])))

            initial_hidden_dim = self._Cells[0]._W_x2i.shape[1]
            indices_to_copy = np.random.randint(initial_hidden_dim, size=(new_hidden_dim - initial_hidden_dim))
            replication_factor = np.bincount(indices_to_copy, minlength=initial_hidden_dim)

            # Growing all the RNN cells
            self._Cells[0].make_cell_wider(new_hidden_dim=new_layer_size[0],
                                           indices_to_copy=indices_to_copy,
                                           replication_factor=replication_factor,
                                           use_noise=use_noise,
                                           use_random_noise=use_random_noise,
                                           is_first_cell=True)

            for idx, cell in enumerate(self._Cells[1:]):
                cell.make_cell_wider(new_hidden_dim=new_layer_size[idx],
                                     indices_to_copy=indices_to_copy,
                                     replication_factor=replication_factor,
                                     use_noise=use_noise,
                                     use_random_noise=use_random_noise,
                                     is_first_cell=False)

            # Growing the parameters of the recurrent net
            self._layer_size = new_layer_size

            student_w = make_weight_wider_at_input(teacher_w=self._W_h2o.data.cpu().numpy(),
                                                   indices_to_copy=indices_to_copy,
                                                   replication_factor=replication_factor)

            # for index in indices_to_copy:
            #     epsilon = noise[index].pop()
            #     student_w[index+initial_hidden_dim] +=epsilon
            #
            # for index in range(initial_hidden_dim):
            #     if(noise[index]):
            #         student_w[index] += noise[index]

            self._W_h2o.data = torch.from_numpy(student_w)

            # Growing the hidden state vectors
            for h in self._h_prev:
                for key, param in h.items():
                    student_b = make_h_wider(teacher_b=param.data.cpu().numpy(),
                                             indices_to_copy=indices_to_copy)
                    h[key].data = torch.from_numpy(student_b).to(self._device)

            if can_make_optimizer_wider == True:
                logging.info("Making Optimizer wider. Previous width = {}, new width = {}".format(
                    "_".join([str(x) for x in self._layer_size]),
                    "_".join([str(x) for x in new_layer_size])))

                self.make_optimizer_wider(new_hidden_dim=new_hidden_dim, indices_to_copy=indices_to_copy,
                                          replication_factor=replication_factor,
                                          use_noise=use_noise,
                                          use_random_noise=use_random_noise)

    @property
    def layer_size(self):
        return self._layer_size

    def make_optimizer_wider(self, new_hidden_dim, indices_to_copy,
                             replication_factor, use_noise, use_random_noise):
        # Note that this function is written specifically with this network in mind

        param_indices_to_widen_in_input_dim = [0, 3, 7, 11, 15]
        param_indices_to_widen_in_output_dim = [2, 3, 6, 7, 10, 11, 14, 15]
        param_indices_to_widen_in_bias = [4, 5, 7, 8, 9, 12, 13, 16]

        for index, (param, state) in enumerate(self.optimizer.state_dict()["state"].items()):
            # print(index)
            for key in ["exp_avg", "exp_avg_sq"]:
                # print(state[key].shape)
                if index in param_indices_to_widen_in_input_dim:
                    state[key] = torch.from_numpy(
                        make_weight_wider_at_input(teacher_w=state[key].data.cpu().numpy(),
                                                   indices_to_copy=indices_to_copy,
                                                   replication_factor=replication_factor)) \
                        .to(self._device)
                if index in param_indices_to_widen_in_output_dim:
                    state[key] = torch.from_numpy(
                        make_weight_wider_at_output(teacher_w=state[key].data.cpu().numpy(),
                                                    indices_to_copy=indices_to_copy,
                                                    use_noise=use_noise,
                                                    use_random_noise=use_random_noise
                                                    )) \
                        .to(self._device)

                if index in param_indices_to_widen_in_bias:
                    state[key] = torch.from_numpy(
                        make_bias_wider(teacher_b=state[key].data.cpu().numpy(),
                                        indices_to_copy=indices_to_copy)) \
                        .to(self._device)
                # print(state[key].shape)

            # print("=============")

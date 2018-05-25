"""Implementation of an LSTM Cell."""
import numpy as np
import torch

from myTorch.memory.LSTMCell import LSTMCell
from myTorch.projects.overfeeding.utils.net2net import make_bias_wider, make_weight_wider_at_output, \
    make_weight_wider_at_input, generate_noise_for_input


class ExpandableLSTMCell(LSTMCell):
    """Implementation of an LSTM Cell based on https://arxiv.org/pdf/1308.0850.pdf"""

    def __init__(self, device, input_size, hidden_size, layer_norm=False, chrono_init = False, t_max = 10):
        """Initializes an LSTM Cell.

        Args:
            device: torch device object.
            input_size: int, size of the input vector.
            hidden_size: int, LSTM hidden layer dimension.
            layer_norm: bool, if True, applies layer normalization.
        """

        super(ExpandableLSTMCell, self).__init__(device=device, input_size=input_size,
                                       hidden_size=hidden_size, layer_norm=layer_norm,
                                       chrono_init=chrono_init, t_max=t_max)

    def reset_hidden_randomly(self, batch_size):
        """Resets the hidden state randomly for testing the `make_cell_wider` operation."""

        hidden = {}
        hidden["h"] = torch.Tensor(np.random.random((batch_size, self._hidden_size))).to(self._device)
        hidden["c"] = torch.Tensor(np.random.random((batch_size, self._hidden_size))).to(self._device)
        return hidden

    def make_cell_wider(self, new_hidden_dim, indices_to_copy, replication_factor, is_first_cell=False):
        """
            Method to make the recurrent net wider by growing the original hidden dim to the
            size new_hidden_dim
            Args:
                 new_hidden_dim: Size of the hidden dim of the recurrent net after making it wider
        """
        weights_to_widen_in_output_dim = ["_W_x2i", "_W_x2f", "_W_x2o", "_W_x2c"]
        weights_to_widen_in_input_and_output_dim = ["_W_h2i", "_W_h2f", "_W_h2o", "_W_h2c"]
        biases_to_widen = ["_W_c2i", "_b_i", "_W_c2f", "_b_f", "_W_c2o", "_b_o", "_b_c", ]
        if (not is_first_cell):
            weights_to_widen_in_input_and_output_dim = weights_to_widen_in_input_and_output_dim \
                                                       + weights_to_widen_in_output_dim
            weights_to_widen_in_output_dim = []

        self._hidden_size = new_hidden_dim
        for name, param in self.named_parameters():
            if (name in weights_to_widen_in_output_dim):
                student_w1 = make_weight_wider_at_output(teacher_w=param.data.cpu().numpy(),
                                                         indices_to_copy=indices_to_copy,
                                                         replication_factor=replication_factor)
                param.data = torch.from_numpy(student_w1)

            elif (name in biases_to_widen):
                student_b = make_bias_wider(teacher_b=param.data.cpu().numpy(),
                                            indices_to_copy=indices_to_copy,
                                                         replication_factor=replication_factor)
                param.data = torch.from_numpy(student_b)
            elif (name in weights_to_widen_in_input_and_output_dim):
                student_w = make_weight_wider_at_input(teacher_w=param.data.cpu().numpy(),
                                                       indices_to_copy=indices_to_copy,
                                                       replication_factor=replication_factor)
                student_w = make_weight_wider_at_output(teacher_w=student_w.copy(),
                                                        indices_to_copy=indices_to_copy,
                                                         replication_factor=replication_factor)
                param.data = torch.from_numpy(student_w)
            else:
                print("Error! Found a parameter {} for which rules are not defined".format(name))

"""Implementation of a simple MLP."""
import torch.nn as nn
import torch.nn.functional as F
from myTorch.utils import get_activation
from myTorch.utils import Model


class MLP(Model):
    """A simple MLP class."""

    def __init__(self, num_hidden_layers=2, hidden_layer_size=[10, 10], activation="relu",
                 input_dim=100, output_dim=10):
        """Initialize an MLP object.

        Args:
            num_hidden_layers: int, number of hidden layers.
            hidden_layer_size: list of int, number of units per hidden layer.
            activation: str, name of the activation function.
            input_dim: int, dimension of the input.
            output_dim: int, dimension of the output.
        """

        super(MLP, self).__init__()
        self._num_hidden_layers = num_hidden_layers
        self._hidden_layer_size = hidden_layer_size
        self._activation = get_activation(activation)
        self._input_dim = input_dim
        self._output_dim = output_dim

        self._fc_layers = []
        self._fc_layers.append(nn.Linear(input_dim, hidden_layer_size[0]))
        for i in range(1, num_hidden_layers):
            self._fc_layers.append(nn.Linear(hidden_layer_size[i-1], hidden_layer_size[i]))
        self._fc_layers.append(nn.Linear(hidden_layer_size[-1], output_dim))

        self._module_list = nn.ModuleList(self._fc_layers)

    def forward(self, input):
        """Forward prop for the MLP.

        Args:
            input: torch Tensor with shape (batch_size, input_dim).

        Returns:
            Output of the model which is a torch Tensor with shape (batch_size, output_dim).
        """

        x = input

        for i in range(len(self._fc_layers)-1):
            x = self._activation(self._fc_layers[i](x))
        x = self._fc_layers[-1](x)

        return x

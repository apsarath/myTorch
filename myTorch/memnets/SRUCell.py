import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import math

class SRUCell(nn.Module):

    def __init__(self, device, input_size, hidden_size, phi_size=256, r_size=64, activation="tanh",
                 A=[0, 0.5, 0.9, 0.99, 0.999]): 
        super(SRUCell, self).__init__()

        self._device = device

        self._input_size = input_size
        self._hidden_size = hidden_size
        self._phi_size = phi_size
        self._r_size = r_size
        self._A = A
        self._n_alpha = len(A)
        self._mu_size = self._phi_size * self._n_alpha

        self._mu2r   = nn.Linear(self._mu_size, self._r_size)
        self._xr2phi = nn.Linear(self._input_size + self._r_size, self._phi_size)
        self._mu2o   = nn.Linear(self._mu_size, self._hidden_size)

        self._A_mask = torch.Tensor([x for x in(A) for i in range(phi_size)]).view(1, -1).to(self._device)
        self._init_weight()
        

    def forward(self, input, last_hidden):
        r = F.relu(self._mu2r(last_hidden["mu"]))
        phi = F.relu(self._xr2phi(torch.cat((input, r), 1)))
        mu_next = self._muphi2mu(last_hidden["mu"], phi)
        hidden = {}
        hidden["h"] = F.relu(self._mu2o(mu_next))
        hidden["mu"] = mu_next
        return hidden

    def _muphi2mu(self, mu, phi):
        phi_tile = phi.repeat(1, self._n_alpha)
        mu = torch.mul(self._A_mask, mu) + torch.mul((1-self._A_mask), phi_tile)
        return mu

    def reset_hidden(self, batch_size):
        hidden = {}
        hidden["mu"] = torch.Tensor(np.zeros((batch_size, self._mu_size))).to(self._device)
        return hidden

    def _init_weight(self):
        for name, params in self.named_parameters():
            if 'weight' in name:
                init.xavier_uniform(params, init.calculate_gain('relu'))
            else:
                init.constant(params, 0)

import torch
import torch.nn as nn
import numpy as np
import math


import myTorch
from myTorch.utils import act_name


class FlatMemoryCell(nn.Module):

    def __init__(self, device, input_size, hidden_size, memory_size=900, k=16, activation="tanh"):
        
        super(FlatMemoryCell, self).__init__()

        self._device = device

        self.input_size = input_size
        self.hidden_size = hidden_size
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "sigmoid":
            self.activation = torch.sigmoid 
        
        self.W_i2h = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_h2h = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_h = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

        self.memory_size = memory_size
        self.k = k
        assert math.sqrt(self.memory_size*self.k).is_integer()
        sqrt_memk = int(math.sqrt(self.memory_size*self.k))
        self.hm2v_alpha = nn.Linear(self.memory_size + hidden_size, sqrt_memk)
        self.hm2v_beta = nn.Linear(self.memory_size + hidden_size, sqrt_memk)
        self.hm2alpha = nn.Linear(self.memory_size + hidden_size, self.k)
        self.hm2beta = nn.Linear(self.memory_size + hidden_size, self.k)

        self.hm2h = nn.Linear(self.memory_size+hidden_size, hidden_size)

        
    def forward(self, input, last_hidden):
        
        
        c_input = torch.cat((input, last_hidden["h_internal"]), 1)
        W_h = torch.cat((self.W_i2h, self.W_h2h), 0)
        pre_hidden = torch.add(torch.mm(c_input, W_h), self.b_h)
        h = self.activation(pre_hidden)

        # Flat memory equations
        alpha = self.hm2alpha(torch.cat((last_hidden["h_internal"],last_hidden["memory"]),1))
        beta = self.hm2beta(torch.cat((last_hidden["h_internal"],last_hidden["memory"]),1))

        v_alpha = self.hm2v_alpha(torch.cat((last_hidden["h_internal"],last_hidden["memory"]),1))
        v_alpha = torch.bmm(v_alpha.unsqueeze(2), v_alpha.unsqueeze(1)).view(-1, self.k, self.memory_size)
        phi_alpha = torch.nn.functional.cosine_similarity(v_alpha, last_hidden["memory"].expand_as(v_alpha), dim=-1)
        add_memory = (phi_alpha*alpha).unsqueeze(2)*v_alpha

        v_beta = self.hm2v_beta(torch.cat((last_hidden["h_internal"],last_hidden["memory"]),1))
        v_beta = torch.bmm(v_beta.unsqueeze(2), v_beta.unsqueeze(1)).view(-1, self.k, self.memory_size)
        phi_beta = torch.nn.functional.cosine_similarity(v_beta, last_hidden["memory"].expand_as(v_beta), dim=-1)
        forget_memory = (phi_beta*beta).unsqueeze(2)*v_beta

        hidden = {}
        hidden["h_internal"] = h
        hidden["memory"] = last_hidden["memory"] + torch.sum(add_memory - forget_memory, dim=1)
        hidden["h"] = self.hm2h(torch.cat((hidden["h_internal"],hidden["memory"]),1))
        return hidden

    def reset_hidden(self):
        hidden = {}
        hidden["h_internal"] = torch.Tensor(np.zeros((1, self.hidden_size))).to(self._device)
        hidden["memory"] = torch.Tensor(np.zeros((1, self.memory_size))).to(self._device)

        return hidden

    def reset_parameters(self):

        nn.init.xavier_normal(self.W_i2h, gain=nn.init.calculate_gain(act_name(self.activation)))
        nn.init.orthogonal(self.W_h2h)
        nn.init.constant(self.b_h, 0)

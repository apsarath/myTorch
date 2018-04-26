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
        
        self.memory_size = memory_size
        self.k = k
        assert math.sqrt(self.memory_size*self.k).is_integer()
        sqrt_memk = int(math.sqrt(self.memory_size*self.k))
        self.hm2v_alpha = nn.Linear(self.memory_size + hidden_size, 2 * sqrt_memk)
        self.hm2v_beta = nn.Linear(self.memory_size + hidden_size, 2 * sqrt_memk)
        self.hm2alpha = nn.Linear(self.memory_size + hidden_size, self.k)
        self.hm2beta = nn.Linear(self.memory_size + hidden_size, self.k)

        self.hmi2h = nn.Linear(self.memory_size+hidden_size+self.input_size, hidden_size)

        
    def forward(self, input, last_hidden):
        c_input = torch.cat((input, last_hidden["h"], last_hidden["memory"]), 1)
        h = self.activation(self.hmi2h(c_input))

        # Flat memory equations
        alpha = self.hm2alpha(torch.cat((h,last_hidden["memory"]),1)).clone()
        beta = self.hm2beta(torch.cat((h,last_hidden["memory"]),1)).clone()

        u_alpha = self.hm2v_alpha(torch.cat((h,last_hidden["memory"]),1)).chunk(2,dim=1)
        v_alpha = torch.bmm(u_alpha[0].unsqueeze(2), u_alpha[1].unsqueeze(1)).view(-1, self.k, self.memory_size)
        add_memory = alpha.unsqueeze(2)*v_alpha

        u_beta = self.hm2v_beta(torch.cat((h,last_hidden["memory"]),1)).chunk(2, dim=1)
        v_beta = torch.bmm(u_beta[0].unsqueeze(2), u_beta[1].unsqueeze(1)).view(-1, self.k, self.memory_size)
        forget_memory = beta.unsqueeze(2)*v_beta

        hidden = {}
        hidden["memory"] = last_hidden["memory"]*(torch.mean(add_memory, dim=1) - torch.mean(forget_memory, dim=1))
        hidden["h"] = h
        return hidden

    def reset_hidden(self, batch_size):
        hidden = {}
        hidden["h"] = Variable(torch.Tensor(np.zeros((batch_size, self.hidden_size))))
        hidden["memory"] = Variable(torch.Tensor(np.zeros((batch_size, self.memory_size))))
        if self.use_gpu==True:
            hidden["h"] = hidden["h"].cuda()
            hidden["memory"] = hidden["memory"].cuda()
        return hidden

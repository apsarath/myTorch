import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math



class FlatMemoryCellV2(nn.Module):

    def __init__(self, device, input_size, hidden_size, memory_size=64, k=4, activation="tanh", use_relu=False, layer_norm=False): 
        super(FlatMemoryCellV2, self).__init__()

        self._device = device

        self.input_size = input_size
        self.hidden_size = hidden_size
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "sigmoid":
            self.activation = torch.sigmoid 
        
        self.memory_size = memory_size
        self.k = k
        self._use_relu =  use_relu
        self._layer_norm = layer_norm

        assert math.sqrt(self.memory_size*self.k).is_integer()
        sqrt_mem = int(math.sqrt(self.memory_size))
        
        self.hm2v_alpha = nn.Linear(self.memory_size + hidden_size, 2 * k * sqrt_mem)
        self.hm2v_beta = nn.Linear(self.memory_size + hidden_size, 2 * k * sqrt_mem)
        self.hm2alpha = nn.Linear(self.memory_size + hidden_size, self.k)
        self.hm2beta = nn.Linear(self.memory_size + hidden_size, self.k)

        if self._layer_norm:
            self._ln_h = nn.LayerNorm(hidden_size)

        #self.hmi2h = torch.nn.LSTM(input_size=self.memory_size + hidden_size + self.input_size, hidden_size=hidden_size)
        self.hmi2h = nn.Linear(self.memory_size + hidden_size + self.input_size, hidden_size)
        #self.hmi2r = nn.Linear(self.memory_size + hidden_size + self.input_size, self.k)
        
    def _opt_relu(self, x):
        if self._use_relu:
            return F.relu(x)
        else:
            return x

    def _opt_layernorm(self, x):
        if self._layer_norm:
            return self._ln_h(x)
        else:
            return x

    def forward(self, input, last_hidden):
        hidden = {}
        c_input = torch.cat((input, last_hidden["h"], last_hidden["memory"]), 1)
        #r_t = self.hmi2r(c_input)
        #c_input = torch.cat((input, last_hidden["h"], r_t), 1)
        
        #h, hidden["lstm_cell_h"] = self.hmi2h(c_input.unsqueeze(0), last_hidden["lstm_cell_h"])
        #h = h.squeeze(0)
        h = F.relu(self._opt_layernorm(self.hmi2h(c_input)))

        # Flat memory equations
        alpha = self._opt_relu(self.hm2alpha(torch.cat((h,last_hidden["memory"]),1))).clone()
        beta = self._opt_relu(self.hm2beta(torch.cat((h,last_hidden["memory"]),1))).clone()

        u_alpha = self.hm2v_alpha(torch.cat((h,last_hidden["memory"]),1)).chunk(2,dim=1)
        u_alpha = [u.chunk(self.k, dim=1) for u in u_alpha] 
        v_alpha = []
        for i in range(self.k):
            v_alpha.append(torch.bmm(u_alpha[0][i].unsqueeze(2), u_alpha[1][i].unsqueeze(1)).view(-1, self.memory_size))
        v_alpha = torch.stack(v_alpha, dim=1)
        v_alpha = self._opt_relu(v_alpha)
        v_alpha = torch.nn.functional.normalize(v_alpha, p=5, dim=2, eps=1e-12)
        add_memory = alpha.unsqueeze(2)*v_alpha

        u_beta = self.hm2v_beta(torch.cat((h,last_hidden["memory"]),1)).chunk(2, dim=1)
        u_beta = [u.chunk(self.k, dim=1) for u in u_beta]
        v_beta = []
        for i in range(self.k):
            v_beta.append(torch.bmm(u_beta[0][i].unsqueeze(2), u_beta[1][i].unsqueeze(1)).view(-1, self.memory_size))
        v_beta = torch.stack(v_beta, dim=1)
        v_beta = self._opt_relu(v_beta)
        v_beta = torch.nn.functional.normalize(v_beta, p=5, dim=2, eps=1e-12)
        forget_memory = beta.unsqueeze(2)*v_beta

        hidden["memory"] = last_hidden["memory"] + torch.mean(add_memory-forget_memory, dim=1)
        hidden["h"] = h
        return hidden

    def reset_hidden(self, batch_size):
        hidden = {}
        hidden["h"] = torch.Tensor(np.zeros((batch_size, self.hidden_size))).to(self._device)
        hidden["memory"] = torch.Tensor(np.zeros((batch_size, self.memory_size))).to(self._device)
        #hidden["lstm_cell_h"] = (torch.Tensor(np.zeros((1, batch_size, self.hidden_size))).to(self._device), 
        #                        torch.Tensor(np.zeros((1, batch_size, self.hidden_size))).to(self._device))
        return hidden

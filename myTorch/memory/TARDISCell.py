import torch
import torch.nn as nn
import numpy as np
from myTorch.utils.gumbel_softmax import gumbel_softmax, gumbel_sigmoid

from torch.autograd import Variable


class TARDISCell(nn.Module):

    def __init__(self, input_size, hidden_size, micro_state_size=50, num_mem_cells=10, activation=None, use_gpu=False):

        super(LSTMCell, self).__init__()

        self.use_gpu = use_gpu

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.micro_state_size = micro_state_size
        self.num_mem_cells = num_mem_cells

        self.W_x2i = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_h2i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_c2i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))

        self.W_x2f = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_h2f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_c2f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))

        self.W_x2o = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_h2o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_c2o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        self.W_x2c = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_h2c = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(hidden_size))

        self.memory = []
        self.W_m = nn.Parameter(torch.Tensor(hidden_size, micro_state_size))
        self.b_m = nn.Parameter(torch.Tensor(micro_state_size))

        self.w_a_h = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.w_b_h = nn.Parameter(torch.Tensor(hidden_size, 1))

        self.w_a_x = nn.Parameter(torch.Tensor(input_size, 1))
        self.w_b_x = nn.Parameter(torch.Tensor(input_size, 1))

        self.w_a_r = nn.Parameter(torch.Tensor(micro_state_size, 1))
        self.w_b_r = nn.Parameter(torch.Tensor(micro_state_size, 1))
        self.alpha_beta_params = [self.w_a_h, self.w_b_h, self.w_a_x, self.w_b_x, self.w_a_r, self.w_b_r]
        self.reset_parameters()

    def forward(self, input, last_hidden):

        self.W_i = torch.cat((self.W_x2i, self.W_h2i, self.W_c2i), 0)
        self.W_f = torch.cat((self.W_x2f, self.W_h2f, self.W_c2f), 0)
        self.W_o = torch.cat((self.W_x2o, self.W_h2o, self.W_c2o), 0)
        self.W_c = torch.cat((self.W_x2c, self.W_h2c), 0)

        c_input = torch.cat((input, last_hidden["h"], last_hidden["c"]), 1)
        i = torch.sigmoid(torch.mm(c_input, self.W_i) + self.b_i)
        f = torch.sigmoid(torch.mm(c_input, self.W_f) + self.b_f)

        cp_input = torch.cat((input, last_hidden["h"]), 1)
        cp = torch.tanh(torch.mm(cp_input, self.W_c) + self.b_c)
        c = f * last_hidden["c"] + i * cp

        o_input = torch.cat((input, last_hidden["h"], c), 1)
        o = torch.sigmoid(torch.mm(o_input, self.W_o) + self.b_o)

        h = o * torch.tanh(c)

        # memory construction
        curr_micro_state = torch.mm(h, self.W_m) + self.b_m
        if len(self.memory) < self.num_mem_cells:
            self.memory.append(curr_micro_state)
        else:
            # dot product with memory so far.
            memory_matrix = torch.stack(self.memory, axis=1)
            logits = torch.squeeze(torch.bmm(memory_matrix, curr_micro_state), dim=-1)
            sampled_locations = gumbel_softmax(logits)
            sampled_mirco_state = torch.add(torch.bmm(memory_matrix, sampled_locations), 1)

            alpha = torch.mm(h, self.w_a_h) + torch.mm(input, self.w_a_x) + torch.mm(sampled_mirco_state, self.w_a_r)
            alpha = gumbel_sigmoid(alpha)
            beta = torch.mm(h, self.w_b_h) + torch.mm(input, self.w_b_x) + torch.mm(sampled_mirco_state, self.w_b_r)
            beta = gumbel_sigmoid(beta)

        hidden = {}
        hidden["h"] = h
        hidden["c"] = c
        return hidden

    def reset_hidden(self):
        hidden = {}
        hidden["h"] = Variable(torch.Tensor(np.zeros((1, self.hidden_size))))
        hidden["c"] = Variable(torch.Tensor(np.zeros((1, self.hidden_size))))
        if self.use_gpu:
            hidden["h"] = hidden["h"].cuda()
            hidden["c"] = hidden["c"].cuda()
        return hidden

    def reset_parameters(self):

        nn.init.xavier_normal(self.W_x2i)
        nn.init.xavier_normal(self.W_x2f)
        nn.init.xavier_normal(self.W_x2o)
        nn.init.xavier_normal(self.W_x2c)
        nn.init.xavier_normal(self.W_m)

        for param in self.alpha_beta_params:
            nn.init.xavier_normal(param)

        nn.init.orthogonal(self.W_h2i)
        nn.init.orthogonal(self.W_h2f)
        nn.init.orthogonal(self.W_h2o)
        nn.init.orthogonal(self.W_h2c)

        nn.init.orthogonal(self.W_c2i)
        nn.init.orthogonal(self.W_c2f)
        nn.init.orthogonal(self.W_c2o)

        nn.init.constant(self.b_i, 0)
        nn.init.constant(self.b_f, 1)
        nn.init.constant(self.b_o, 0)
        nn.init.constant(self.b_c, 0)
        nn.init.constant(self.b_m, 0)

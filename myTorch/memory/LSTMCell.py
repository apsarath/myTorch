import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable

class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, activation=None, use_gpu=False):
        
        super(LSTMCell, self).__init__()

        self.use_gpu = use_gpu

        self.input_size = input_size
        self.hidden_size = hidden_size
        
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
        o = torch.sigmoid(torch.mm(o_input ,self.W_o) + self.b_o)
        
        h = o*torch.tanh(c)
        
        hidden = {}
        hidden["h"] = h
        hidden["c"] = c 
        return hidden

    def reset_hidden(self):
        hidden = {}
        hidden["h"] = Variable(torch.Tensor(np.zeros((1,self.hidden_size))))
        hidden["c"] = Variable(torch.Tensor(np.zeros((1,self.hidden_size))))
        if self.use_gpu==True:
            hidden["h"] = hidden["h"].cuda()
            hidden["c"] = hidden["c"].cuda()
        return hidden

    def reset_parameters(self):

        nn.init.xavier_normal(self.W_x2i)
        nn.init.xavier_normal(self.W_x2f)
        nn.init.xavier_normal(self.W_x2o)
        nn.init.xavier_normal(self.W_x2c)
        
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

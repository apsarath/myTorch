import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from myTorch.memory import RNNCell, GRUCell, LSTMCell
from myTorch.memnets.FlatMemoryCell import FlatMemoryCell


class RecurrentCartPole(nn.Module):

    def __init__(self, obs_dim, action_dim, device, rnn_type="LSTM"):
        super(self.__class__, self).__init__()

        if isinstance(obs_dim, tuple):
            self._is_obs_image = (len(obs_dim) > 2)
            self._obs_dim = obs_dim[0]
        else:
            self._is_obs_image = False
            self._obs_dim = obs_dim

        self._action_dim = action_dim
        self._rnn_input_size = self._obs_dim
        self._rnn_hidden_size = 30
        self._rnn_type = rnn_type
        self._is_rnn_policy = True
        self._device = device
        if self._rnn_type == "LSTM": 
            self._rnn = LSTMCell(self._device, input_size=self._rnn_input_size, hidden_size=self._rnn_hidden_size)
        elif self._rnn_type == "GRU":
            self._rnn = GRUCell(self._device, input_size=self._rnn_input_size, hidden_size=self._rnn_hidden_size)
        elif self._rnn_type == "FLatMemory":
            self._rnn = FlatMemoryCell(self._device, input_size, hidden_size)

        self._hidden = None

        if self._is_obs_image:
            self._conv1 = nn.Conv2d(self._obs_dim, 16, kernel_size=5, stride=2)
            self._bn1 = nn.BatchNorm2d(16)
            self._conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
            self._bn2 = nn.BatchNorm2d(32)
            self._conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
            self._bn3 = nn.BatchNorm2d(32)
            self._fc1 = nn.Linear(448, 64)

        self._fcv = nn.Linear(self._rnn_hidden_size, 1)
        self._fcp = nn.Linear(self._rnn_hidden_size, self._action_dim)

    def _conv_to_linear(self, obs):
        assert(self._is_obs_image)
        if len(obs.shape) < 4:
            obs = obs.unsqueeze(0)
            x = F.relu(self._bn1(self._conv1(obs)))
            x = F.relu(self._bn2(self._conv2(x)))
            x = F.relu(self._bn3(self._conv3(x)))
            return self._fc1(x.view(x.size(0), -1))

    def reset_hidden(self, batch_size):
        self._hidden = {}
        self._hidden["h"] = torch.zeros(batch_size, self._rnn_hidden_size).to(self._device)
        
        if self._rnn_type == "LSTM":
            self._hidden["c"] = torch.zeros(batch_size, self._rnn_hidden_size).to(self._device)

    def forward(self, obs, update_hidden_state):
        if self._hidden is None:
            self.reset_hidden(obs.shape[0])

        x = self._conv_to_linear(obs) if self._is_obs_image else obs
        hidden_next = self._rnn(x, self._hidden)
        if update_hidden_state:
            self._hidden = hidden_next
        p = self._fcp(hidden_next["h"])
        v = self._fcv(hidden_next["h"])
        return p, v

    def update_hidden(self, mask):
        for k in self._hidden:
            self._hidden[k] = mask.expand(-1,self._rnn_hidden_size) * self._hidden[k]

    def detach_hidden(self):
        for k in self._hidden:
            self._hidden[k] = self._hidden[k].detach()
            #hidden_value = self._hidden[k].data.cpu().numpy()
            #self._hidden[k] =  torch.from_numpy(hidden_value)
        
    @property
    def action_dim(self):
        return self._action_dim

    @property
    def obs_dim(self):
        return self._obs_dim

    @property
    def device(self):
        return self._device

    @property
    def is_rnn_policy(self):
        return self._is_rnn_policy

    def get_attributes(self):
        return (self._obs_dim, self._action_dim, self._device, self._rnn_type)

    def get_params(self):
        return self.state_dict()

    def set_params(self, state_dict):
        self.load_state_dict(state_dict)

    def make_inference_net(self):
        inference_net = self.__class__(*self.get_attributes()).to(self._device)
        inference_net.set_params(self.get_params())
        return inference_net


if __name__=="__main__":
    p, v = RecurrentCartPole(50,10)
    import pdb; pdb.set_trace()


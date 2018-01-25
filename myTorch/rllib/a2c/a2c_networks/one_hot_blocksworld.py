import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from myTorch.utils import my_variable
from myTorch.memory import RNNCell, GRUCell, LSTMCell, TARDISCell


class RecurrentOneHotBlocksWorldMatrix(nn.Module):

	def __init__(self, obs_dim, action_dim, use_gpu=False, rnn_type="LSTM"):
		super(self.__class__, self).__init__()

		self._obs_dim = obs_dim
		self._action_dim = action_dim
		self._use_gpu = use_gpu
		self._rnn_input_size = 32
		self._rnn_hidden_size = 30
		self._rnn_type = rnn_type
		self._is_rnn_policy = True
		if self._rnn_type == "LSTM": 
			self._rnn = LSTMCell(input_size=self._rnn_input_size, hidden_size=self._rnn_hidden_size, use_gpu=use_gpu)
		elif self._rnn_type == "GRU":
			self._rnn = GRUCell(input_size=self._rnn_input_size, hidden_size=self._rnn_hidden_size, use_gpu=use_gpu)
		self._hidden = None

		self._conv1 = nn.Conv2d(self._obs_dim[0], 16, kernel_size=2, stride=1)
		self._bn1 = nn.BatchNorm2d(16)
		self._conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1)
		self._bn2 = nn.BatchNorm2d(32)
		self._conv3 = nn.Conv2d(32, 32, kernel_size=2, stride=1)
		self._bn3 = nn.BatchNorm2d(32)
		self._conv4 = nn.Conv2d(32, 32, kernel_size=4, stride=1)
		self._bn4 = nn.BatchNorm2d(32)
		self._fc1 = nn.Linear(512, self._rnn_input_size)

		self._fcv = nn.Linear(self._rnn_hidden_size, 1)
		self._fcp = nn.Linear(self._rnn_hidden_size, self._action_dim)

	def _conv_to_linear(self, obs):
		if len(obs.shape) < 4:
			obs = obs.unsqueeze(0)
		x = F.relu(self._bn1(self._conv1(obs)))
		x = F.relu(self._bn2(self._conv2(x)))
		x = F.relu(self._bn3(self._conv3(x)))
		x = F.relu(self._bn4(self._conv4(x)))
		return self._fc1(x.view(x.size(0), -1))

	def reset_hidden(self, batch_size):
		self._hidden = {}
		self._hidden["h"] = my_variable(torch.zeros(batch_size, self._rnn_hidden_size), use_gpu=self._use_gpu)
		
		if self._rnn_type == "LSTM":
			self._hidden["c"] = my_variable(torch.zeros(batch_size, self._rnn_hidden_size), use_gpu=self._use_gpu)

	def forward(self, obs, update_hidden_state):
		if self._hidden is None:
			self.reset_hidden(obs.shape[0])

		x = self._conv_to_linear(obs)
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
			hidden_value = self._hidden[k].data.cpu().numpy()
			self._hidden[k] =  my_variable(torch.from_numpy(hidden_value), use_gpu=self._use_gpu)
		
	@property
	def action_dim(self):
		return self._action_dim

	@property
	def obs_dim(self):
		return self._obs_dim

	@property
	def use_gpu(self):
		return self._use_gpu

	@property
	def is_rnn_policy(self):
		return self._is_rnn_policy

	def get_attributes(self):
		return (self._obs_dim, self._action_dim, self._use_gpu, self._rnn_type)

	def get_params(self):
		return self.state_dict()

	def set_params(self, state_dict):
		self.load_state_dict(state_dict)

	def make_inference_net(self):
		inference_net = self.__class__(*self.get_attributes())
		if self._use_gpu == True:
				inference_net.cuda()
		inference_net.set_params(self.get_params())
		return inference_net


if __name__=="__main__":
	p, v = RecurrentCartPole(50,10)
	import pdb; pdb.set_trace()


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from myTorch.utils import my_variable


class RecurrentCartPole(nn.Module):

	def __init__(self, obs_dim, action_dim, use_gpu=False, rnn_type="LSTM"):
		super(self.__class__, self).__init__()

		if isinstance(obs_dim, tuple):
			self._is_obs_image = (len(obs_dim) > 2)
			self._obs_dim = obs_dim[0]
		else:
			self._is_obs_image = False
			self._obs_dim = obs_dim

		self._action_dim = action_dim
		self._use_gpu = use_gpu
		self._rnn_input_size = 20
		self._rnn_hidden_size = 30
		self._rnn_type = rnn_type
		self._is_rnn_policy = True
		if self._rnn_type == "LSTM": 
			self._rnn = torch.nn.LSTMCell(input_size=self._rnn_input_size, hidden_size=self._rnn_hidden_size, bias=True)
		elif self._rnn_type == "GRU":
			self._rnn = torch.nn.GRUCell(input_size=self._rnn_input_size, hidden_size=self._rnn_hidden_size, bias=True)
		self._hidden = None

		if self._is_obs_image:
			self._conv1 = nn.Conv2d(self._obs_dim, 16, kernel_size=5, stride=2)
			self._bn1 = nn.BatchNorm2d(16)
			self._conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
			self._bn2 = nn.BatchNorm2d(32)
			self._conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
			self._bn3 = nn.BatchNorm2d(32)
			self._fc1 = nn.Linear(448, 64)
		else:
			self._fc1 = nn.Linear(self._obs_dim, 64)

		self._fc2 = nn.Linear(64, 64)
		self._fc3 = nn.Linear(64, 64)
		self._fc4 = nn.Linear(64, self._rnn_input_size)

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

	def _rnn_step(self, x):
		if self._rnn_type == "GRU":
			hidden_next = self._rnn(x, self._hidden)
			return (hidden_next, hidden_next)
		elif self._rnn_type == "LSTM":
			rnn_output, hidden_next = self._rnn(x, self._hidden)
			return (rnn_output, hidden_next)

	def _reset_hidden(self, batch_size):
		if self._rnn_type == "GRU":
			self._hidden = my_variable(torch.zeros(batch_size, self._rnn_hidden_size), use_gpu=self._use_gpu)
		elif self._rnn_type == "LSTM":
			self._hidden = (my_variable(torch.zeros(batch_size, self._rnn_hidden_size), use_gpu=self._use_gpu), 
							my_variable(torch.zeros(batch_size, self._rnn_hidden_size), use_gpu=self._use_gpu))

	def forward(self, obs, update_hidden_state):
		if self._hidden is None:
			self._reset_hidden(obs.shape[0])

		x = self._conv_to_linear(obs) if self._is_obs_image else self._fc1(obs)
		x = F.relu(self._fc2(x))
		x = F.relu(self._fc3(x))
		x = F.relu(self._fc4(x))
		rnn_output, hidden_next = self._rnn_step(x)
		if update_hidden_state:
			self._hidden = hidden_next
		p = F.relu(self._fcp(rnn_output))
		v = F.relu(self._fcv(rnn_output))
		return p, v

	def update_hidden(self, mask):
		if self._rnn_type == "GRU":
			self._hidden = mask.expand(-1,self._rnn_hidden_size) * self._hidden
		elif self._rnn_type == "LSTM":
			h, c = self._hidden
			self._hidden = (mask.expand(-1,self._rnn_hidden_size) * h, 
							mask.expand(-1,self._rnn_hidden_size) * c)
	def detach_hidden(self):
		if self._rnn_type == "GRU":
			hidden_value = self._hidden.data.cpu().numpy()
			self._hidden = my_variable(torch.from_numpy(hidden_value), use_gpu=self._use_gpu)
		elif self._rnn_type == "LSTM":
			h_v, c_v = self._hidden[0].data.cpu().numpy(), self._hidden[1].data.cpu().numpy()
			self._hidden = (my_variable(torch.from_numpy(h_v), use_gpu=self._use_gpu),
							my_variable(torch.from_numpy(c_v), use_gpu=self._use_gpu))
		
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
		return (self._obs_dim, self._action_dim, self._use_gpu)

	def get_params(self):
		return self.state_dict()

	def set_params(self, state_dict):
		self.load_state_dict(state_dict)


if __name__=="__main__":
	p, v = RecurrentCartPole(50,10)
	import pdb; pdb.set_trace()


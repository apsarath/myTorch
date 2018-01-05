import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardCartPole(nn.Module):

	def __init__(self, obs_dim, action_dim, use_gpu=False):
		super(self.__class__, self).__init__()

		self._obs_dim = obs_dim[0] if isinstance(obs_dim, tuple) else obs_dim
		self._action_dim = action_dim
		self._use_gpu = use_gpu

		self._fc1 = nn.Linear(self._obs_dim, 64)
		self._fc2 = nn.Linear(64, 64)
		
		self._fcv1 = nn.Linear(64, 64)
		self._fcv2 = nn.Linear(64, 32)
		self._fcv3 = nn.Linear(32, 1)

		self._fcp1 = nn.Linear(64, 64)
		self._fcp2 = nn.Linear(64, 32)
		self._fcp3 = nn.Linear(32, self._action_dim)

	def forward(self, input):
		x = F.relu(self._fc1(input))
		x = F.relu(self._fc2(x))

		v = F.relu(self._fcv1(x))
		v = F.relu(self._fcv2(v))
		v = F.relu(self._fcv3(v))
		
		p = F.relu(self._fcp1(x))
		p = F.relu(self._fcp2(p))
		p = F.relu(self._fcp3(p))
		return p, v

	def softmax(self, inp):
		return F.softmax(inp)

	@property
	def action_dim(self):
		return self._action_dim

	@property
	def obs_dim(self):
		return self._obs_dim

	@property
	def use_gpu(self):
		return self._use_gpu

	def get_attributes(self):
		return (self._obs_dim, self._action_dim, self._use_gpu)

	def get_params(self):
		return self.state_dict()

	def set_params(self, state_dict):
		self.load_state_dict(state_dict)


if __name__=="__main__":
	p, v = FeedForward(50,10)
	import pdb; pdb.set_trace()

if __name__=="__main__":
	x = FeedForward(50,10)
	x1 = FeedForward.make_target_net(x)
	import pdb; pdb.set_trace()



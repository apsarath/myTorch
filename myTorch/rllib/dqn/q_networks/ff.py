import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):

	def __init__(self, obs_dim, action_dim, use_gpu=False):
		super(FeedForward, self).__init__()

		self._obs_dim = obs_dim
		self._action_dim = action_dim
		self._use_gpu = use_gpu

		self._fc1 = nn.Linear(obs_dim, 64)
		self._fc2 = nn.Linear(64, 64)
		self._fc3 = nn.Linear(64, action_dim)

	def forward(self, input):
		x = F.relu(self._fc1(input))
		x = F.relu(self._fc2(x))
		x = self._fc3(x)
		return x

	@property
	def action_dim(self):
		return self._action_dim

	@property
	def use_gpu(self):
		return self._use_gpu

	def get_attributes(self):
		return (self._obs_dim, self._action_dim, self._use_gpu)

	def get_params(self):
		return self.state_dict()

	def set_params(self, state_dict):
		self.load_state_dict(state_dict)

	@staticmethod
	def make_target_net(qnet):
		target_net = FeedForward(*qnet.get_attributes())
		return target_net

if __name__=="__main__":
	x = FeedForward(50,10)
	x1 = FeedForward.make_target_net(x)
	import pdb; pdb.set_trace()



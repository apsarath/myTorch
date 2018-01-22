import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvMazeBase(nn.Module):

	def __init__(self, obs_dim, action_dim, use_gpu=False):
		super(self.__class__, self).__init__()

		self._obs_dim = obs_dim[0] if isinstance(obs_dim, tuple) else obs_dim
		self._action_dim = action_dim
		self._use_gpu = use_gpu
		self._is_rnn_policy = False

		self._conv1 = nn.Conv2d(4, 8, kernel_size=3, stride=1)
		self._bn1 = nn.BatchNorm2d(8)
		self._conv2 = nn.Conv2d(8, 16, kernel_size=2, stride=1)
		self._bn2 = nn.BatchNorm2d(16)
		self._headp = nn.Linear(560, action_dim)
		self._headv = nn.Linear(560, 1)

	def forward(self, x):
		if len(x.shape) < 4:
			x = x.unsqueeze(0)
		x = F.relu(self._bn1(self._conv1(x)))
		x = F.relu(self._bn2(self._conv2(x)))
		p = self._headp(x.view(x.size(0), -1))
		v = self._headv(x.view(x.size(0), -1))
		return p, v

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

	def make_inference_net(self):
		inference_net = self.__class__(*self.get_attributes())
		if self._use_gpu == True:
			inference_net.cuda()
		inference_net.set_params(self.get_params())
		return inference_net

if __name__=="__main__":
	x = FeedForward(50,10)
	x1 = FeedForward.make_target_net(x)
	import pdb; pdb.set_trace()

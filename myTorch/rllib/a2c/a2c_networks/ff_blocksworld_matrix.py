import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardBlocksWorldMatrix(nn.Module):

	def __init__(self, obs_dim, action_dim, use_gpu=False):
		super(self.__class__, self).__init__()
		self._obs_dim = obs_dim
		self._action_dim = action_dim
		self._use_gpu = use_gpu
		self._is_rnn_policy = False

		self._conv1 = nn.Conv2d(self._obs_dim[0], 16, kernel_size=5, stride=2)
		self._bn1 = nn.BatchNorm2d(16)
		self._conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
		self._bn2 = nn.BatchNorm2d(32)
		self._conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
		self._bn3 = nn.BatchNorm2d(32)
		self._fc1 = nn.Linear(288, 100)
		self._fcv = nn.Linear(100, 1)

		self._fcp = nn.Linear(100, self._action_dim)

	def forward(self, obs):
		if len(obs.shape) < 4:
			obs = obs.unsqueeze(0)
		x = F.relu(self._bn1(self._conv1(obs)))
		x = F.relu(self._bn2(self._conv2(x)))
		x = F.relu(self._bn3(self._conv3(x)))
		x = self._fc1(x.view(x.size(0), -1))
		v = self._fcv(x)
		p = self._fcp(x)
		return p, v

	def softmax(self, inp, dim=1):
		return F.softmax(inp, dim=dim)

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
	p, v = FeedForward(50,10)
	import pdb; pdb.set_trace()

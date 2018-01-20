import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlocksWorld(nn.Module):

	def __init__(self, obs_dim, action_dim, use_gpu=False):
		super(self.__class__, self).__init__()

		self._obs_dim = obs_dim
		self._action_dim = action_dim
		self._use_gpu = use_gpu

		self._conv1 = nn.Conv2d(self._obs_dim[0], 16, kernel_size=2, stride=1)
		self._bn1 = nn.BatchNorm2d(16)
		self._conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1)
		self._bn2 = nn.BatchNorm2d(32)
		self._conv3 = nn.Conv2d(32, 32, kernel_size=2, stride=1)
		self._bn3 = nn.BatchNorm2d(32)
		self._conv4 = nn.Conv2d(32, 32, kernel_size=4, stride=1)
		self._bn4 = nn.BatchNorm2d(32)
		self._fc1 = nn.Linear(512,100)
		self._head = nn.Linear(100, self._action_dim)


	def forward(self, x):
		if len(x.shape) < 4:
			x = x.unsqueeze(0)
		x = F.relu(self._bn1(self._conv1(x)))
		x = F.relu(self._bn2(self._conv2(x)))
		x = F.relu(self._bn3(self._conv3(x)))
		x = F.relu(self._bn4(self._conv4(x)))
		x = F.relu(self._fc1(x.view(x.size(0), -1)))
		return self._head(x)


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

	def make_target_net(self, qnet):
		target_net = self.__class__(*qnet.get_attributes())
		if self._use_gpu == True:
			target_net.cuda()
		return target_net

if __name__=="__main__":
	x = FeedForward(50,10)
	x1 = FeedForward.make_target_net(x)
	import pdb; pdb.set_trace()

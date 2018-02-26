import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardCartPole(nn.Module):

	def __init__(self, obs_dim, action_dim, use_gpu=False, is_dueling_arch=False):
		super(self.__class__, self).__init__()

		self._obs_dim = obs_dim[0] if isinstance(obs_dim, tuple) else obs_dim
		self._action_dim = action_dim
		self._use_gpu = use_gpu
		self._is_dueling_arch = is_dueling_arch

		self._fc1 = nn.Linear(self._obs_dim, 100)
		if self._is_dueling_arch:
			self._fcv = nn.Linear(100, 1)
			self._fca = nn.Linear(100, self._action_dim)
		else:
			self._fc2 = nn.Linear(100, self._action_dim)


	def forward(self, input):
		if len(input.shape) < 2:
			input = input.unsqueeze(0)
		x = F.relu(self._fc1(input))
		if self._is_dueling_arch:
			state_val = self._fcv(x)
			adv_vals = self._fca(x)
			return adv_vals.sub_(torch.mean(adv_vals, dim=1, keepdim=True)) + state_val
		else:
			return self._fc2(x)


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



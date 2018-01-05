import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


import myTorch
from myTorch.utils import my_variable

class A2CAgent(object):

	def __init__(self, a2cnet, optimizer, numpy_rng, discount_rate=0.99, grad_clip=None):
		self._a2cnet = a2cnet
		self._optimizer = optimizer
		self._numpy_rng = numpy_rng
		self._discount_rate = discount_rate
		self._grad_clip = grad_clip
		self._loss = nn.SmoothL1Loss()


	def sample_action(self, obs, legal_moves=None, is_training=True):
		obs = my_variable(torch.from_numpy(obs), use_gpu=self._a2cnet.use_gpu)
		pvals, vvals = self._a2cnet.forward(obs)

		if legal_moves is None:
				legal_moves = torch.zeros_like(pvals)

		pvals = self._a2cnet.softmax(pvals + legal_moves)
		actions = torch.multinomial(pvals, 1)
		log_pvals = torch.log(torch.gather(pvals, 1, actions))
		return actions, log_pvals, vvals

	def train_step(self, minibatch):

		self._optimizer.zero_grad()


		for key in minibatch:
			minibatch[key] = my_variable(torch.from_numpy(minibatch[key]), use_gpu=self._qnet.use_gpu, volatile=True, requires_grad=False)


		loss = self._loss(predicted_action_values, action_value_targets)

		loss.backward()

		if self._grad_clip[0] is not None:
			for param in self._qnet.parameters():
				param.grad.data.clamp_(self._grad_clip[0], self._grad_clip[1])

		self._optimizer.step()

		return loss.data[0]

	def save(self, dname):

		fname = os.path.join(dname, "a2cnet.p")
		torch.save(self._qnet.state_dict(), fname)

		fname = os.path.join(dname, "optimizer.p")
		torch.save(self._optimizer.state_dict(), fname)


	def load(self, dname):

		fname = os.path.join(dname, "a2cnet.p")
		self._qnet.load_state_dict(torch.load(fname))

		fname = os.path.join(dname, "optimizer.p")
		self._optimizer.load_state_dict(torch.load(fname))


	@property
	def action_dim(self):
		return self._a2cnet.action_dim

if __name__=="__main__":

	from myTorch.rllib.a2c.a2c_networks import *
	qnet = FeedForwardCartPole(4,2)
	agent = A2CAgent(qnet,None,None)
	sample_obs = np.ones((10,4), dtype=np.float32)
	action, log_pvals, vvals = agent.sample_action(sample_obs)
	import pdb;pdb.set_trace()

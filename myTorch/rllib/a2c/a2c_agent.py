import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


import myTorch
from myTorch.utils import my_variable

class A2CAgent(object):

	def __init__(self, a2cnet, optimizer, numpy_rng, ent_coef = 1.0, vf_coef = 1.0,  discount_rate=0.99, grad_clip=None):
		self._a2cnet = a2cnet
		self._optimizer = optimizer
		self._numpy_rng = numpy_rng
		self._discount_rate = discount_rate
		self._ent_coef = ent_coef
		self._vf_coef = vf_coef
		self._grad_clip = grad_clip
		self._loss = nn.SmoothL1Loss()


	def sample_action(self, obs, legal_moves=None, is_training=True):
		obs = my_variable(torch.from_numpy(obs), use_gpu=self._a2cnet.use_gpu)
		pvals, vvals = self._a2cnet.forward(obs)

		if legal_moves is None:
				legal_moves = torch.ones_like(pvals)

		pvals = self._a2cnet.softmax(pvals*legal_moves)
		entropies = -torch.sum(pvals * torch.log(pvals), dim=1, keepdim=True)
		actions = torch.multinomial(pvals, 1)
		log_taken_pvals = torch.log(torch.gather(pvals, 1, actions))
		return actions, log_taken_pvals, vvals, entropies

	def train_step(self, minibatch):

		self._optimizer.zero_grad()

		num_steps = len(minibatch.values()[0])
		batch_size = minibatch["episode_dones"][0].shape[0]

		R = [minibatch["episode_dones"][-1] * minibatch["vvals"][-1]]
		for t in reversed(range(num_steps-1)):
			R.append((minibatch["rewards"][t] + self._discount_rate* R[-1]) * ( 1  - minibatch["episode_dones"][t] ) + \
				(minibatch["vvals"][t] * minibatch["episode_dones"][t]))

		# convert to Torch Variables
		R = my_variable(torch.from_numpy(torch.stack(R[::-1], dim=1), use_gpu=self._qnet.use_gpu, volatile=True, requires_grad=False))
		for v_types in ["vvals", "log_taken_pvals", "entropies"]:
			 minibatch[key] = torch.stack(minibatch[key], dim=1)
			 minibatch[key] = my_variable(torch.from_numpy(minibatch[key]), use_gpu=self._qnet.use_gpu)

		## reduce mean for everything
		pg_loss, val_loss, entropy_loss = 0, 0, 0
		for t in range(num_steps-1):
			pg_loss += torch.mean((R[:,t] -  minibatch["vvals"][:,t]).detach()* minibatch["log_taken_pvals"][:,t])
			val_loss += torch.nn.functional.mse_loss(minibatch["vvals"][:,t], R[:,t].detach())
			entropy_loss -= torch.mean(minibatch["entropies"][:,t])

		# reduce

		loss = pg_loss + self._vf_coef * val_loss + self._ent_coef * entropy_loss
		loss = loss / (num_steps-1)

		loss.backward()

		if self._grad_clip[0] is not None:
			for param in self._qnet.parameters():
				param.grad.data.clamp_(self._grad_clip[0], self._grad_clip[1])

		self._optimizer.step()

		return pg_loss.data[0], val_loss.data[0], entropy_loss.data[0]

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

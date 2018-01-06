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


	def sample_action(self, obs, dones=None, rnn_policy=True, legal_moves=None, is_training=True):
		obs = my_variable(torch.from_numpy(obs).type(torch.FloatTensor), use_gpu=self._a2cnet.use_gpu)
		if self._a2cnet.is_rnn_policy:
			self._a2cnet.update_hidden(dones)

		pvals, vvals = self._a2cnet.forward(obs)

		if legal_moves is None:
				legal_moves = torch.ones_like(pvals)

		pvals = torch.nn.functional.softmax(pvals*legal_moves, dim=1)
		entropies = -torch.sum(pvals * torch.log(pvals), dim=1)
		if is_training:
			actions = torch.multinomial(pvals, 1).detach()
		else:
			actions = torch.max(pvals, dim=1)[1]

		log_taken_pvals = torch.log(torch.gather(pvals, 1, actions))
		return actions.data.cpu().numpy().squeeze(1), log_taken_pvals.squeeze(1), vvals.squeeze(1), entropies

	def train_step(self, minibatch):

		self._optimizer.zero_grad()

		num_steps = len(minibatch.values()[0])
		batch_size = minibatch["episode_dones"][0].shape[0]
		
		# Convert from numpy arrays into torch Variables
		for data_key in ["episode_dones","rewards"]:
			minibatch[data_key] = my_variable(torch.from_numpy(np.stack(minibatch[data_key])).type(torch.FloatTensor), use_gpu=self._a2cnet.use_gpu) 
	
		R = [minibatch["vvals_step_plus_one"]]
		for t in reversed(range(num_steps)):
			R.append(minibatch["rewards"][t] + self._discount_rate* R[-1]*(1  - minibatch["episode_dones"][t]))

		R = R[::-1]

		# Compute loss
		pg_loss, val_loss, entropy_loss = 0, 0, 0
		for t in range(num_steps):
			pg_loss += torch.mean((R[t] -  minibatch["vvals"][t]).detach()* minibatch["log_taken_pvals"][t])
			val_loss += torch.nn.functional.mse_loss(minibatch["vvals"][t], R[t].detach())
			entropy_loss -= torch.mean(minibatch["entropies"][t])

		loss = pg_loss + self._vf_coef * val_loss + self._ent_coef * entropy_loss
		loss = loss / num_steps
		
		loss.backward()

		if self._grad_clip[0] is not None:
			for param in self._a2cnet.parameters():
				param.grad.data.clamp_(self._grad_clip[0], self._grad_clip[1])

		self._optimizer.step()

		return pg_loss.data[0], val_loss.data[0], entropy_loss.data[0]

	def save(self, dname):

		fname = os.path.join(dname, "a2cnet.p")
		torch.save(self._a2cnet.state_dict(), fname)

		fname = os.path.join(dname, "optimizer.p")
		torch.save(self._optimizer.state_dict(), fname)


	def load(self, dname):

		fname = os.path.join(dname, "a2cnet.p")
		self._a2cnet.load_state_dict(torch.load(fname))

		fname = os.path.join(dname, "optimizer.p")
		self._optimizer.load_state_dict(torch.load(fname))


	@property
	def action_dim(self):
		return self._a2cnet.action_dim

if __name__=="__main__":

	from myTorch.rllib.a2c.a2c_networks import *
	a2cnet = FeedForwardCartPole(4,2)
	agent = A2CAgent(a2cnet,None,None)
	sample_obs = np.ones((10,4), dtype=np.float32)
	action, log_pvals, vvals = agent.sample_action(sample_obs)
	import pdb;pdb.set_trace()

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


import myTorch
from myTorch.utils import my_variable

class A2CAgent(object):

	def __init__(self, a2cnet, optimizer, discount_rate=0.99, 
		grad_clip = None, 
		target_net_update_freq=10000,
		epsilon_start=1, epsilon_end=0.1, epsilon_end_t = 1e5, learn_start=50000):

		self._a2cnet = a2cnet
		self._optimizer = optimizer
		self._discount_rate = discount_rate
		self._grad_clip = grad_clip
		self._target_net_update_freq = target_net_update_freq
		self._epsilon_start = epsilon_start
		self._epsilon_end = epsilon_end
		self._epsilon_end_t = epsilon_end_t
		self._learn_start = learn_start

		self._target_a2cnet = self._a2cnet.make_target_net(self._a2cnet)
		self._loss = nn.SmoothL1Loss()


	def sample_action(self, obs, legal_moves=None, step=None):

		obs = my_variable(torch.from_numpy(obs), use_gpu=self._a2cnet.use_gpu)
		pvals, vvals = self._a2cnet.forward(obs)

		if legal_moves is None:
			legal_moves = my_variable(torch.ones_like(pvals))

		pvals = self._a2cnet.softmax(pvals + legal_moves)
		actions = torch.multinomial(input=pvals, num_samples=1).squeeze()
		log_pvals = torch.log(torch.gather(pvals, 1, actions))
		return actions, log_pvals, vvals.squeeze()

	def train_step(self, actions_list, log_pvals_list, vvals_list, rewards_list, ep_continue_list):

		self._optimizer.zero_grad()
		R = np.torch([vvals_list[-1][i] for i in range(config.num_envs) if ep_continue_list[-1][i] else 0])
		loss_vf = 0
		loss_pol = 0
		for t in reversed(range(config.num_bptt_steps-1)):
			R = R*self._discount_rate*ep_continue_list[t] + torch.array(rewards_list[t])
			loss_vf += (R - torch.array(vvals_list[t]))**2
			loss_pol += (R - torch.array(vvals_list[t])).detach()*(log_pvals_list[t])
		loss_vf.backward()
		loss_pol.backward()
		
		#if self._grad_clip is not None:
		#	for param in self._q_net.parameters():
		#		param.grad.data.clamp_(self._grad_clip[0], self._grad_clip[1])

		self._optimizer.step()

		return loss.data.numpy()

	def update_target_net(self):

		self._target_a2cnet.set_params(self._a2cnet.get_params)

if __name__=="__main__":

	a2cnet = FeedForward(50,10)
	agent = A2CAgent(a2cnet)
	action, qval = agent.sample_action(np.ones((1,50), dtype="float32"), is_training=False)
	#print(actions)
	import pdb
	pdb.set_trace()




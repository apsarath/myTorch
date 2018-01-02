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


	def sample_action(self, obs, legal_moves=None, epsilon=0, step=None, is_training=True):

		if is_training:
			epsilon = (self._epsilon_end + max(0., (self._epsilon_start - self._epsilon_end)*
				(self._epsilon_end_t - max(0., step - self._learn_start))
				/ self._epsilon_end_t))

		
		obs = my_variable(torch.from_numpy(obs), use_gpu=self._a2cnet.use_gpu)
		pvals, vvals = self._a2cnet.forward(obs)
		
		if legal_moves is None:
			legal_moves = my_variable(torch.ones_like(pvals))
		
		pvals = self._a2cnet.softmax(pvals + legal_moves)
		actions = torch.multinomial(input=pvals, num_samples=1).squeeze()
		log_pvals = torch.log(torch.gather(pvals, 1, actions))
		return actions, log_pvals, vvals.squeeze()

	def train_step(self, minibatch):

		self._optimizer.zero_grad()

		for key in minibatch:
			if key in ["observations_tp1", "legal_moves_tp1"]:
				minibatch[key] = my_variable(torch.from_numpy(minibatch[key]), use_gpu=self._a2cnet.use_gpu, volatile=True, requires_grad=False)
			else:
				minibatch[key] = my_variable(torch.from_numpy(minibatch[key]), use_gpu=self._a2cnet.use_gpu)

		predicted_action_values = self._a2cnet.forward(minibatch["observations"])
		predicted_action_values *= minibatch["actions"]
		predicted_action_values = torch.sum(predicted_action_values, dim=1)

		next_step_action_values = self._target_a2cnet.forward(minibatch["observations_tp1"])
		next_step_action_values += minibatch["legal_moves_tp1"]
		next_step_best_actions_values = torch.max(next_step_action_values, dim=1).detach()

		action_value_targets = minibatch["rewards"] + self._discount_rate * self.next_step_best_actions_values * minibatch["pcontinues"]

		loss = self._loss(self.predicted_action_values, self.action_value_targets)

		loss.backward()

		if self._grad_clip is not None:
			for param in self._q_net.parameters():
				param.grad.data.clamp_(self._grad_clip[0], self._grad_clip[1])

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




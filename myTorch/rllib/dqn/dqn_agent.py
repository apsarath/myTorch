import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


import myTorch
from myTorch.utils import my_variable

class DQNAgent(object):

	def __init__(self, qnet, optimizer, numpy_rng, discount_rate=0.99, 
		grad_clip = None,
		is_double_dqn = False,
		target_net_soft_update = False, 
		target_net_update_freq=10000,
		target_net_update_fraction=0.05,
		epsilon_start=1, epsilon_end=0.1, epsilon_end_t = 1e5, learn_start=50000):

		self._qnet = qnet
		self._is_double_dqn = is_double_dqn
		self._optimizer = optimizer
		self._numpy_rng = numpy_rng
		self._discount_rate = discount_rate
		self._grad_clip = grad_clip
		self._target_net_soft_update = target_net_soft_update
		self._target_net_update_freq = target_net_update_freq
		self._target_net_update_fraction = target_net_update_fraction
		self._epsilon_start = epsilon_start
		self._epsilon_end = epsilon_end
		self._epsilon_end_t = epsilon_end_t
		self._learn_start = learn_start

		self._target_qnet = self._qnet.make_target_net(self._qnet)
		self._loss = nn.SmoothL1Loss()


	def sample_action(self, obs, legal_moves=None, epsilon=0, step=None, is_training=True):


		if is_training:
			epsilon = (self._epsilon_end + max(0., (self._epsilon_start - self._epsilon_end)*
				(self._epsilon_end_t - max(0., step - self._learn_start))
				/ self._epsilon_end_t))

		if legal_moves is None:
			legal_moves = np.zeros(self._qnet.action_dim)

		if self._numpy_rng.random_sample() < epsilon:
			actions = np.where(legal_moves==0)[0]
			r = self._numpy_rng.randint(0, len(actions))
			return actions[r], None

		# TO DO : check the need to convert to float tensor explictly.
		obs = my_variable(torch.from_numpy(obs).type(torch.FloatTensor), use_gpu=self._qnet.use_gpu)
		qvals = self._qnet.forward(obs).data.cpu().numpy().flatten()
		qvals = qvals + legal_moves
		best_action = np.argmax(qvals)
		qval = max(qvals)
		return best_action, qval

	def train_step(self, minibatch):

		self._optimizer.zero_grad()


		for key in minibatch:
			if key in ["observations_tp1", "legal_moves_tp1"]:
				minibatch[key] = my_variable(torch.from_numpy(minibatch[key]), use_gpu=self._qnet.use_gpu, volatile=True, requires_grad=False)
			else:
				minibatch[key] = my_variable(torch.from_numpy(minibatch[key]), use_gpu=self._qnet.use_gpu)

		predicted_action_values = self._qnet.forward(minibatch["observations"])
		predicted_action_values *= minibatch["actions"]
		predicted_action_values = torch.sum(predicted_action_values, dim=1)
		
		next_step_target_net_action_values = self._target_qnet.forward(minibatch["observations_tp1"])
		next_step_target_net_action_values += minibatch["legal_moves_tp1"]

		# Double dqn changes
		if self._is_double_dqn:
			# get Q values of the "next_state" from the curr_net
			next_step_curr_net_action_values = self._qnet.forward(minibatch["observations_tp1"])
			next_step_curr_net_action_values += minibatch["legal_moves_tp1"]

			# argmax over the Curr-net's Q values to select greedy action.
			next_step_best_actions = torch.max(next_step_curr_net_action_values, dim=1)[1].detach().unsqueeze(1)

			# choose the Q-values of the target-net for the greedy action chosen from the prev line.
			next_step_best_actions_values = torch.gather(next_step_target_net_action_values, 1, \
												 next_step_best_actions).squeeze(1)
		else:
			next_step_best_actions_values = torch.max(next_step_target_net_action_values, dim=1)[0].detach()

		action_value_targets = minibatch["rewards"] + self._discount_rate * next_step_best_actions_values * minibatch["pcontinues"]

		loss = self._loss(predicted_action_values, action_value_targets)

		loss.backward()

		if self._grad_clip[0] is not None:
			for param in self._qnet.parameters():
				param.grad.data.clamp_(self._grad_clip[0], self._grad_clip[1])

		self._optimizer.step()

		return loss.data[0]

	def update_target_net(self):

		if self._target_net_soft_update == False:

			self._target_qnet.set_params(self._qnet.get_params())

		else:

			target_params = self._target_qnet.get_params()
			current_params = self._qnet.get_params()
			for key in target_params.keys():
				target_params[key] = (1-self._target_net_update_fraction)*target_params[key] + self._target_net_update_fraction*current_params[key]
			self._target_qnet.set_params(target_params)

	def save(self, dname):

		fname = os.path.join(dname, "qnet.p")
		torch.save(self._qnet.state_dict(), fname)

		fname = os.path.join(dname, "target_qnet.p")
		torch.save(self._target_qnet.state_dict(), fname)

		fname = os.path.join(dname, "optimizer.p")
		torch.save(self._optimizer.state_dict(), fname)


	def load(self, dname):

		fname = os.path.join(dname, "qnet.p")
		self._qnet.load_state_dict(torch.load(fname))

		fname = os.path.join(dname, "target_qnet.p")
		self._target_qnet.load_state_dict(torch.load(fname))

		fname = os.path.join(dname, "optimizer.p")
		self._optimizer.load_state_dict(torch.load(fname))


	@property
	def action_dim(self):

		return self._qnet.action_dim

if __name__=="__main__":

	qnet = FeedForward(50,10)
	agent = DQNAgent(qnet)
	action, qval = agent.sample_action(np.ones((1,50), dtype="float32"),is_training=False)
	#print(actions)
	import pdb
	pdb.set_trace()




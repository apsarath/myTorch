import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


import myTorch
from myTorch.rllib.dqn.q_networks import FeedForward
from myTorch.utils import my_variable

class DQNAgent(object):

	def __init__(self, qnet, discount_rate=0.99, learning_rate=1e-3, target_net_update_freq=10000,
		epsilon_start=1, epsilon_end=0.1, epsilon_end_t = 1e5, learn_start=50000):

		self._qnet = qnet
		self._discount_rate = discount_rate
		self._learning_rate = learning_rate
		self._target_net_update_freq = target_net_update_freq
		self._epsilon_start = epsilon_start
		self._epsilon_end = epsilon_end
		self._epsilon_end_t = epsilon_end_t
		self._learn_start = learn_start

		self._target_qnet = FeedForward.make_target_net(self._qnet)


	def sample_action(self, obs, legal_moves=None, epsilon=0, step=None, is_training=True):

		if is_training:
			epsilon = (self._epsilon_end + max(0., (self._epsilon_start - self._epsilon_end)*
				(self._epsilon_end_t - max(0., step - self._learn_start))
				/ self._epsilon_end_t))

		if legal_moves is None:
			legal_moves = np.zeros(self._qnet.action_dim)

		if np.random.random_sample() < epsilon:
			actions = np.where(legal_moves==0)[0]
			r = np.random.randint(0, len(actions))
			return actions[r], None

		obs = my_variable(torch.from_numpy(obs), use_gpu=self._qnet.use_gpu)
		qvals = self._qnet.forward(obs).data.numpy().flatten()
		qvals = qvals + legal_moves
		best_action = np.argmax(qvals)
		qval = max(qvals)
		return best_action, qval

	def train_step(self, minibatch, steps_done, next_target_upd):

		for key in minibatch:
			minibatch[key] = my_variable(torch.from_numpy(minibatch[key]), use_gpu=self._qnet.use_gpu)

		predicted_action_values = self._qnet.forward(minibatch["observations"])
		predicted_action_values *= minibatch["actions"]
		predicted_action_values = torch.sum(predicted_action_values, dim=1)

		next_step_action_values = self._target_qnet.forward(minibatch["observations_tp1"])
		next_step_action_values += minibatch["legal_moves_tp1"]
		next_step_best_actions_values = torch.max(next_step_action_values, dim=1)

		action_value_targets = minibatch["rewards"] + self._discount_rate * self.next_step_best_actions_values * minibatch["pcontinues"]

		loss = nn.SmoothL1Loss(self.predicted_action_values, self.action_value_targets)




if __name__=="__main__":

	qnet = FeedForward(50,10)
	agent = DQNAgent(qnet)
	action, qval = agent.sample_action(np.ones((1,50), dtype="float32"),is_training=False)
	#print(actions)
	import pdb
	pdb.set_trace()




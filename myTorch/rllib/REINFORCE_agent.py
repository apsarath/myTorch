import torch
from torch.autograd import Variable
import torch.nn.utils as utils
import math



class REINFORCE(object):

	def __init__(self, model, optimizer, gamma, alg_param, use_gpu=False):

		self.model = model
		self.optimizer = optimizer
		self.gamma = gamma
		self.use_gpu = use_gpu

		self.use_etpy = alg_param["use_etpy"]
		if self.use_etpy == True:
			self.etpy_coeff = alg_param["etpy_coeff"]

		self.init_record()

	def init_record(self):

		self.log_probs = []
		self.entropies = []
		self.rewards = []

	def compute_action_prob(self, state):

		probs = self.model(state)
		return probs

	def select_action(self, state):

		probs = self.compute_action_prob(state)
		action = probs.multinomial().data	# 1 X 1 matrix
		prob = probs[:, action[0,0]].view(1, -1) # again 1 X 1 matrix
		log_prob = prob.log()
		entropy = - (probs*probs.log()).sum()

		self.log_probs.append(log_prob)
		self.entropies.append(entropy)

		return action[0], log_prob, entropy

	def best_action(self, state):

		probs = self.compute_action_prob(state)
		m, am = torch.max(probs, 1)
		am = am.view(1,-1).data
		return am[0]

	def record_reward(self, reward):

		self.rewards.append(reward)

	def update_parameters(self):
		
		R = torch.zeros(1, 1)
		loss = 0
		for i in reversed(range(len(self.rewards))):
			R = (self.gamma * R) + (self.rewards[i])
			loss = loss - (self.log_probs[i]*self.cudify(Variable(R).expand_as(self.log_probs[i]))).sum()
			if self.use_etpy:
			 	loss = loss - (self.etpy_coeff*self.entropies[i]).sum()
		loss = loss / len(self.rewards)
		
		self.optimizer.zero_grad()
		loss.backward()
		utils.clip_grad_norm(self.model.parameters(), 40)
		self.optimizer.step()
		self.init_record()

	def cudify(self, var):
		if self.use_gpu==True:
			return var.cuda()
		else:
			return var

	def save(self, dname):
		return

	def load(self, dname):
		return
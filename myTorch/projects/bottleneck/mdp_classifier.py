import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from myTorch.utils import my_variable
from myTorch.memory import RNNCell, GRUCell, LSTMCell, TARDISCell

class MDPCLassifier(nn.Module):
	def __init__(self, action_dim, obs_dim, num_clusters, num_rewards=1, use_gpu=False, rnn_type="LSTM", grad_clip=[None, None]):
		super(self.__class__, self).__init__()
	
		self._use_gpu = use_gpu
		self._action_dim = action_dim
		self._obs_dim = obs_dim
		self._num_clusters = num_clusters
		self._num_rewards = num_rewards
		self._rnn_hidden_size = 256
		self._embedding_dim = 20
		self._rnn_input_size = self._embedding_dim
		self._max_vocab_size = 100
		self._max_num_steps = 30
		self._pad = 0
		self._optimizer = None
		self._grad_clip = grad_clip

		self._rnn_type = rnn_type
		rnn_cell = LSTMCell if self._rnn_type== "LSTM" else GRUCell
		self._obs_cell = rnn_cell(input_size=self._rnn_input_size, hidden_size=self._rnn_hidden_size, use_gpu=self._use_gpu)
		self._rnn = { "obs" : self._obs_cell }


		self._input_embedding = nn.Embedding(self._max_vocab_size, self._embedding_dim, sparse=False, padding_idx=self._pad)
		self._action_emb = nn.Linear(self._action_dim, 64)
		self._fc1 = nn.Linear(self._rnn_hidden_size, 64)
		self._fc2 = nn.Linear(2*64, 64)
		self._fc3 = nn.Linear(2*64, 64)

		self._fc4 = nn.Linear(64, self._num_clusters)
		self._fc5 = nn.Linear(64, self._num_rewards)
		self._hidden = None

		self._reward_loss = nn.MSELoss()
		self._cluster_loss = nn.CrossEntropyLoss()

	def reset_hidden(self, batch_size):
		self._hidden = {}
		for k in self._rnn:
			self._hidden[k] = {}
			self._hidden[k]["h"] = my_variable(torch.zeros(batch_size, self._rnn_hidden_size), use_gpu=self._use_gpu)
			if self._rnn_type == "LSTM":
				self._hidden[k]["c"] = my_variable(torch.zeros(batch_size, self._rnn_hidden_size), use_gpu=self._use_gpu)

	def _encode(self, input, data_type):
		self.reset_hidden(input.shape[0])
		hidden_next = self._hidden[data_type]
		encoded_emb = hidden_next['h']
		input_mask = input.eq(self._pad).type(torch.cuda.FloatTensor)
		invert_mask = 1 - input_mask
		input_emb = self._input_embedding(input)
		for i in range(self._max_num_steps):
			hidden_next = self._rnn[data_type](input_emb[:,i,:], hidden_next)
			encoded_emb = (invert_mask[:,i]).unsqueeze(1)*hidden_next['h'] + input_mask[:, i].unsqueeze(1)*encoded_emb
		return encoded_emb
		

	def predict_cluster_id_rewards(self, inputs):
		obs, actions = inputs["obs"], inputs["actions"]
		if len(obs.shape) < 2:
			obs = obs.unsqueeze(0)
		if len(actions.shape) < 2:
			actions = actions.unsqueeze(0)
 
		obs_emb = self._fc1(self._encode(obs, "obs"))
		action_emb = self._action_emb(actions)
		common_repr = torch.cat((obs_emb, action_emb), dim=1)

		cluster_logits = self._fc4(F.relu(self._fc2(common_repr)))
		#cluster_pvals = torch.nn.functional.softmax(cluster_logits, dim=1)
		cluster_ids = torch.max(cluster_logits, dim=1)[1]

		reward_logits = self._fc5(F.relu(self._fc3(common_repr)))
		#reward_pvals = torch.nn.functional.softmax(reward_logits, dim=1)
		#rewards = torch.max(reward_pvals, dim=1)[1]
		
		return cluster_logits, cluster_ids, reward_logits
		#return cluster_pvals, cluster_ids.data.cpu().numpy(), reward_pvals, rewards.data.cpu().numpy()

	def get_params(self):
		return self.state_dict()

	def set_params(self, state_dict):
		self.load_state_dict(state_dict)

	@property
	def use_gpu(self):
		return self._use_gpu

	def compute_loss(self, minibatch, optimizer, is_training):
		if self._optimizer is None:
			self._optimizer = optimizer

		if is_training: optimizer.zero_grad()
		cluster_logits, cluster_ids, reward_logits = self.predict_cluster_id_rewards(minibatch)
		reward_loss = self._reward_loss(reward_logits, minibatch["rewards"])
		cluster_loss = self._cluster_loss(cluster_logits, minibatch["cluster_ids"])
		total_loss = 10*reward_loss + 10*cluster_loss

		cluster_accuracy = torch.mean(torch.eq(cluster_ids, minibatch["cluster_ids"]).type(torch.cuda.FloatTensor))*100
		cluster_accuracy = cluster_accuracy.data.cpu().numpy()[0]
		if not is_training:
			return cluster_accuracy, total_loss.data.cpu().numpy()[0]
			#return total_loss.data.cpu().numpy()[0]

		total_loss.backward()
		#print "train reward loss : {}, cluster loss : {}, acc : {}".format(reward_loss.data.cpu().numpy()[0], cluster_loss.data.cpu().numpy()[0], cluster_accuracy)

		if self._grad_clip[0] is not None:
			for param in self.parameters():
				param.grad.data.clamp_(self._grad_clip[0], self._grad_clip[1])

		optimizer.step()
		return total_loss

	def load(self, folder_name):
		fname = os.path.join(folder_name, "net.p")
		self.load_state_dict(torch.load(fname))

		fname = os.path.join(folder_name, "optimizer.p")
		self._optimizer.load_state_dict(torch.load(fname))

	def save(self, folder_name):
		fname = os.path.join(folder_name, "net.p")
		torch.save(self.state_dict(), fname)

		fname = os.path.join(folder_name, "optimizer.p")
		torch.save(self._optimizer.state_dict(), fname)

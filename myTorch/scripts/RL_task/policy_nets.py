import torch
import torch.nn as nn
import torch.nn.functional as F

class feed_forward(nn.Module):

	def __init__(self, hidden_size, num_inputs, action_space):

		super(feed_forward, self).__init__()
		self.hidden_size = hidden_size
		self.action_space = action_space
		self.num_outputs = action_space

		self.linear1 = nn.Linear(num_inputs, hidden_size)
		self.linear2 = nn.Linear(hidden_size, self.num_outputs)

	def forward(self, inputs):

		x = inputs
		#print x
		x = F.relu(self.linear1(x))
		action_scores = self.linear2(x)
		return F.softmax(action_scores)
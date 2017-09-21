import numpy 

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.autograd as autograd



import myTorch
from myTorch.model import Recurrent
from myTorch.task.copy_task import *

from config import *

import argparse

parser = argparse.ArgumentParser(description='Algorithm Learning Task')
parser.add_argument('--config', type=str, default="copy_task_RNN", help='config name')
args = parser.parse_args()

config = eval(args.config)()

model = Recurrent(9, 256, 9, output_activation = "sigmoid")

data_gen = CopyDataGen(num_bits = config.num_bits, min_len = config.min_len, max_len = config.max_len)

print config.l_rate

optimizer = optim.SGD(model.parameters(), lr=config.l_rate, momentum=config.momentum)
criteria = nn.BCELoss()


for step in range(0, config.max_steps):

	data = data_gen.next()
	seqloss = []
	for i in range(0, data["seqlen"]):
		
		#x = Variable(torch.randn(1,9))
		#y = Variable(torch.randn(1,9))
		x = Variable(torch.from_numpy(numpy.asarray([data['x'][i]])), requires_grad=True)
		y = Variable(torch.from_numpy(numpy.asarray([data['y'][i]])))
		for j in range(0,1000):
			#optimizer.zero_grad()

			
			
			mask = data['mask'][i]
			#print "j", j
			loss = (x-y).pow(2).sum()
			#print output
			#print y
			#loss = criteria(output, y)
			#print loss
			#step_loss = loss
			

			#seqloss.append(step_loss)
			#autograd.backward(seqloss)
			

			loss.backward()
			print loss.data[0]

			x.data -= 0.01 * x.grad.data
			x.grad.data.zero_()
			#print y.grad
			#print x.grad
			#print model.b_o.grad
			optimizer.step()

			#print "hi"
			#for param in model.parameters():
			#	print param.grad.sum()
			#print model.b_o.grad
			model.reset_hidden()
			optimizer.zero_grad()
			#break
		break
	break
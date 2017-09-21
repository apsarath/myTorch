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

from myTorch.utils.logging import Logger

logger = Logger("/mnt/data/sarath/output/copyRNN")

parser = argparse.ArgumentParser(description='Algorithm Learning Task')
parser.add_argument('--config', type=str, default="copy_task_RNN", help='config name')
args = parser.parse_args()

config = eval(args.config)()

model = Recurrent(9, 256, 9, output_activation = None)

data_gen = CopyDataGen(num_bits = config.num_bits, min_len = config.min_len, max_len = config.max_len)

print config.l_rate

optimizer = optim.RMSprop(model.parameters(), lr=config.l_rate, momentum=config.momentum)
criteria = nn.BCEWithLogitsLoss()

print model.Cell.num_parameters()
print model.num_parameters()


average = []
for step in range(0, config.max_steps):

	data = data_gen.next()

	#print data["seqlen"]
	#print data["mask"]
	#print data["mask"][-1]
	#print data["x"]
	#print data["y"]

	for j in range(0, 1):
		seqloss = 0
		for i in range(0, data["datalen"]):
			#print "i",i
			
			x = Variable(torch.from_numpy(numpy.asarray([data['x'][i]])))
			y = Variable(torch.from_numpy(numpy.asarray([data['y'][i]])))
			#print x
			#print y
			#print data["mask"][i]
			mask = float(data["mask"][i])
			#print mask	

			optimizer.zero_grad()

				
				
			output = model(x)
			#print torch.sigmoid(output)
			loss = criteria(output, y)
			#print loss.data[0]
			seqloss += (loss*mask)
				
			#seqloss.append(step_loss)
		seqloss /= sum(data["mask"])
		average.append(seqloss.data[0])
		logval = sum(average)/len(average)
		logger.log_scalar("loss", logval, step+1)
		seqloss.backward()
		#seqloss.backward()

		optimizer.step()

		model.reset_hidden()
		
	#break

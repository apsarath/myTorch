import numpy 
import argparse

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.autograd as autograd

import myTorch
from myTorch.model import Recurrent
from myTorch.task.copy_task import *
from myTorch.utils.logging import Logger
from myTorch.utils.experiment import Experiment
from myTorch.utils import MyContainer

from config import *

parser = argparse.ArgumentParser(description='Algorithm Learning Task')
parser.add_argument('--config', type=str, default="copy_task_RNN", help='config name')
parser.add_argument('--rdir', type=str, default=None, help='directory to resume')
args = parser.parse_args()


config = eval(args.config)()



logger = None
if config.use_tflogger==True:
	logger = Logger(config.tflogdir)

torch.manual_seed(config.rseed)


model = Recurrent(config.input_size, config.output_size, num_layers = config.num_layers, layer_size=config.layer_size, mname=config.model, output_activation = None, use_gpu=config.use_gpu)

if config.use_gpu==True:
	model.cuda()


data_gen = CopyDataGen(num_bits = config.num_bits, min_len = config.min_len, max_len = config.max_len)

optimizer = config.optim_algo(model.parameters(), lr=config.l_rate, momentum=config.momentum)

criteria = nn.BCEWithLogitsLoss()


trainer = MyContainer()
trainer.ex_seen = 0
trainer.average_bce = []


e = Experiment(model, config, optimizer, trainer, data_gen, logger)

if args.rdir != None:
	e.resume("current", args.rdir)


for step in range(e.trainer.ex_seen, e.config.max_steps):

	data = e.data_gen.next()
	seqloss = 0

	for i in range(0, data["datalen"]):
		
		x = Variable(torch.from_numpy(numpy.asarray([data['x'][i]])))
		y = Variable(torch.from_numpy(numpy.asarray([data['y'][i]])))
		if config.use_gpu == True:
			x = x.cuda()
			y = y.cuda()
		mask = float(data["mask"][i])

		e.optimizer.zero_grad()

		output = e.model(x)
		loss = criteria(output, y)
		seqloss += (loss*mask)
	

	seqloss /= sum(data["mask"])
	#print seqloss.data[0]
	e.trainer.average_bce.append(seqloss.data[0])
	running_average = sum(e.trainer.average_bce)/len(e.trainer.average_bce)
	print running_average
	if e.config.use_tflogger == True:
		logger.log_scalar("loss", running_average, step+1)
	
	seqloss.backward()

	for param in e.model.parameters():
		param.grad.data.clamp_(e.config.grad_clip[0], e.config.grad_clip[1])

	e.optimizer.step()

	e.model.reset_hidden()
	e.trainer.ex_seen += 1

	if e.trainer.ex_seen%10000 == 0:
		e.save("current")
		

import argparse, math, os
import numpy as np
import gym
from gym import wrappers

import torch
from torch.autograd import Variable
import torch.nn.utils as utils

import myTorch
from myTorch.RL.REINFORCE_agent import REINFORCE
from myTorch.utils.logging import Logger
from myTorch.utils.experiment import Experiment
from myTorch.utils import MyContainer

from config import *
from policy_nets import feed_forward

parser = argparse.ArgumentParser(description='REINFORCE')
parser.add_argument('--config', type=str, default="cartpole_reinforce", help='config name')
parser.add_argument('--rdir', type=str, default=None, help='directory to resume')
args = parser.parse_args()

config = eval(args.config)()

env_name = config.env_name
env = gym.make(env_name)

logger = None
logger_id = False
if config.use_tflogger==True:
	logger = Logger(config.tflogdir)
	logger_id = True

torch.manual_seed(config.rseed)

if config.display:
	env = wrappers.Monitor(env, config.video_folder, force=True)


model = feed_forward(config.hidden_size, env.observation_space.shape[0], env.action_space.n, config.use_gpu)
if config.use_gpu==True:
	model.cuda()

optimizer = config.optim_algo(model.parameters(), lr=config.l_rate, momentum=config.momentum)

agent = REINFORCE(model, optimizer, config.gamma, config.alg_param, config.use_gpu)

trainer = MyContainer()
trainer.episodes_done = 0
trainer.curr_ereward = []
trainer.curr_elen = []
if config.validate == True:
	trainer.val_step = []
	trainer.val_elen = []
	trainer.val_ereward = []


e = Experiment(model, config, optimizer, trainer, data_iter=None, logger=logger_id, RL_agent = agent)

if args.rdir != None:
	e.resume("current", args.rdir)

for i_episode in range(e.trainer.episodes_done, e.config.num_episodes):
	state = Variable(torch.Tensor([env.reset()]))
	if e.config.use_gpu == True:
		state = state.cuda()
	ereward = 0
	elen = 0
	for t in range(e.config.num_steps):
		action, log_prob, entropy = e.RL_agent.select_action(state)
		action = action.cpu()

		#print action
		next_state, reward, done, _ = env.step(action.numpy()[0])

		e.RL_agent.record_reward(reward)
		ereward += reward
		state = Variable(torch.Tensor([next_state]))
		if e.config.use_gpu == True:
			state = state.cuda()

		elen+=1

		if done:
			break

	e.RL_agent.update_parameters()
	e.trainer.curr_ereward.append(ereward)
	e.trainer.curr_elen.append(elen)

	if e.config.use_tflogger == True:
		logger.log_scalar("current_reward", ereward, e.trainer.episodes_done+1)
		logger.log_scalar("average_reward", sum(e.trainer.curr_ereward)/len(e.trainer.curr_ereward), e.trainer.episodes_done+1)
		logger.log_scalar("current_epilen", elen, e.trainer.episodes_done+1)
		logger.log_scalar("average_elen", sum(e.trainer.curr_elen)/len(e.trainer.curr_elen), e.trainer.episodes_done+1)

	e.trainer.episodes_done += 1

	if e.trainer.episodes_done%1000 == 0:
		e.save("current")

	if e.config.validate==True:
		if e.trainer.episodes_done%e.config.validate_freq == 0:

			venv_name = e.config.env_name
			venv = gym.make(venv_name)

			v_elen = []
			v_ereward = []
			for j in range(0, e.config.val_num_episodes):

				state = Variable(torch.Tensor([venv.reset()]))
				if e.config.use_gpu == True:
					state = state.cuda()
				ereward = 0
				elen = 0
				for t in range(e.config.val_num_steps):
					action = e.RL_agent.best_action(state)
					action = action.cpu()

					next_state, reward, done, _ = venv.step(action.numpy()[0])

					ereward += reward
					state = Variable(torch.Tensor([next_state]))
					if e.config.use_gpu == True:
						state = state.cuda()

					elen+=1

					if done:
						break
				v_elen.append(elen)
				v_ereward.append(ereward)
			e.trainer.val_step.append(e.trainer.episodes_done)
			e.trainer.val_elen.append(sum(v_elen)/len(v_elen))
			e.trainer.val_ereward.append(sum(v_ereward)/len(v_ereward))
			if e.config.use_tflogger == True:
				logger.log_scalar("val_average_elen", e.trainer.val_elen[-1], e.trainer.val_step[-1] )
				logger.log_scalar("val_average_reward", e.trainer.val_ereward[-1], e.trainer.val_step[-1] )


env.close()
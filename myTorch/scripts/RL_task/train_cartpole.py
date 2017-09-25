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
if config.use_tflogger==True:
	logger = Logger(config.tflogdir)

torch.manual_seed(config.rseed)
env.seed(config.rseed)
np.random.seed(config.rseed)



if config.display:
	env = wrappers.Monitor(env, config.video_folder, force=True)



trainer = MyContainer()
trainer.episodes_done = 0
trainer.curr_ereward = []


model = feed_forward(config.hidden_size, env.observation_space.shape[0], env.action_space.n)

optimizer = config.optim_algo(model.parameters(), lr=config.l_rate, momentum=config.momentum)


e = Experiment(model, config, optimizer, trainer, data_gen=None, logger=logger)

if args.rdir != None:
	e.resume("current", args.rdir)

agent = REINFORCE(e.model, e.optimizer, e.config.gamma, e.config.alg_param, e.config.use_gpu)


for i_episode in range(e.trainer.episodes_done, e.config.num_episodes):
	state = Variable(torch.Tensor([env.reset()]))
	if e.config.use_gpu == True:
		state = state.cuda()
	#print state
	ereward = 0
	for t in range(e.config.num_steps):
		action, log_prob, entropy = agent.select_action(state)
		action = action.cpu()

		next_state, reward, done, _ = env.step(action.numpy()[0])

		agent.record_reward(reward)
		ereward += reward
		state = Variable(torch.Tensor([next_state]))
		if e.config.use_gpu == True:
			state = state.cuda()

		if done:
			break

	agent.update_parameters()
	e.trainer.curr_ereward.append(ereward)

	if e.config.use_tflogger == True:
		e.logger.log_scalar("current_reward", ereward, e.trainer.episodes_done+1)
		e.logger.log_scalar("average_reward", sum(e.trainer.curr_ereward)/len(e.trainer.curr_ereward), e.trainer.episodes_done+1)

	e.trainer.episodes_done += 1



env.close()
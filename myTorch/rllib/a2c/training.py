#!/usr/bin/env python

import os
import math
import numpy as np
import argparse

import torch

import myTorch
from myTorch.environment import get_batched_env
from myTorch.utils import modify_config_params, one_hot, RLExperiment, get_optimizer, my_variable
from myTorch.rllib.a2c.a2c_networks import *

from myTorch.rllib.a2c.config import *
from myTorch.rllib.a2c import A2CAgent
from myTorch.utils import MyContainer
from myTorch.utils.logging import Logger

parser = argparse.ArgumentParser(description="A2C Training")
parser.add_argument('--config', type=str, default="cartpole", help="config name")
parser.add_argument('--base_dir', type=str, default=None, help="base directory")
parser.add_argument('--config_params', type=str, default="default", help="config params to change")
parser.add_argument('--exp_desc', type=str, default="default", help="additional desc of exp")
args = parser.parse_args()


def train_a2c_agent():
	assert(args.base_dir)
	config = eval(args.config)()
	if args.config_params != "default":
		modify_config_params(config, args.config_params)


	train_dir = os.path.join(args.base_dir, config.train_dir, config.exp_name, config.env_name, 
		"{}__{}".format(args.config_params, args.exp_desc))

	logger_dir = os.path.join(args.base_dir, config.logger_dir, config.exp_name, config.env_name,
		"{}__{}".format(args.config_params, args.exp_desc))


	experiment = RLExperiment(config.exp_name, train_dir, config.backup_logger)
	experiment.register_config(config)

	torch.manual_seed(config.seed)
	numpy_rng = np.random.RandomState(seed=config.seed)

	env = get_batched_env(config.env_name, config.num_env, config.seed)
	experiment.register_env(env)

	a2cnet = get_a2cnet(config.env_name, env.obs_dim, env.action_dim, use_gpu=config.use_gpu, policy_type=config.policy_type)

	if config.use_gpu == True:
		a2cnet.cuda()

	optimizer = get_optimizer(a2cnet.parameters(), config)

	agent = A2CAgent(a2cnet, 
			 		optimizer, 
			 		numpy_rng,
					ent_coef = config.ent_coef,
					vf_coef = config.vf_coef,
			 		discount_rate=config.discount_rate, 
			 		grad_clip = [config.grad_clip_min, config.grad_clip_max])
	experiment.register_agent(agent)


	logger = None
	if config.use_tflogger==True:
		logger = Logger(logger_dir)
	experiment.register_logger(logger)

	tr = MyContainer()
	tr.train_reward = [[],[]]
	tr.train_episode_len = [[],[]]
	tr.pg_loss = [[],[]]
	tr.val_loss = [[],[]]
	tr.entropy_loss = [[],[]]
	tr.first_val = [[],[]]
	tr.test_reward = [[],[]]
	tr.test_episode_len = [[],[]]
	tr.iterations_done = 0
	tr.global_steps_done = 0
	tr.episodes_done = 0


	experiment.register_trainer(tr)

	if not config.force_restart:
		if experiment.is_resumable("current"):
			print("resuming the experiment...")
			experiment.resume("current")
	else:
		experiment.force_restart("current")

	num_iterations = config.global_num_steps / (config.num_env * config.num_steps_per_upd)


	obs, legal_moves = env.reset()

	for i in xrange(tr.iterations_done, num_iterations):
		
		print("iterations done: {}".format(tr.iterations_done))

		update_dict = {"log_taken_pvals":[],
					   "vvals":[],
					   "entropies":[],
					   "rewards":[],
					   "episode_dones":[]}

		for t in range(config.num_steps_per_upd):

			actions, log_taken_pvals, vvals, entropies = agent.sample_action(obs, dones=update_dict["episode_dones"], is_training=True)
			obs, legal_moves, rewards, episode_dones = env.step(actions)

			update_dict["log_taken_pvals"].append(log_taken_pvals)
			update_dict["vvals"].append(vvals)
			update_dict["entropies"].append(entropies)
			update_dict["rewards"].append(my_variable(torch.from_numpy(rewards).type(torch.FloatTensor), use_gpu=config.use_gpu))
			update_dict["episode_dones"].append(my_variable(torch.from_numpy(episode_dones.astype(np.float32)).type(torch.FloatTensor), use_gpu=config.use_gpu))

		_, _, update_dict["vvals_step_plus_one"], _ = agent.sample_action(obs, 
																		dones=update_dict["episode_dones"], 
																		is_training=True, 
																		update_agent_state=False)
		pg_loss, val_loss, entropy_loss = agent.train_step(update_dict)
		print "pg_loss : {}, val_loss : {}, entropy_loss : {}".format(pg_loss, val_loss, entropy_loss )

		tr.iterations_done+=1
		tr.global_steps_done = tr.iterations_done*config.num_env*config.num_steps_per_upd
	
	inference_prep(config, agent)
	if math.fmod(tr.global_steps_done, config.save_freq) == 0:
			experiment.save("current")
		

def inference_prep(config, test_agent):
	test_env = get_batched_env(config.env_name, 1, config.seed)
	obs, legal_moves = test_env.reset()
	rewards, episode_lens = [], []
	for i in range(100):
		done, total_reward, episode_len = False, 0 ,0
		obs, legal_moves = test_env.reset()
		test_agent.reset_agent_state(1)
		while not done:
			actions, log_taken_pvals, vvals, entropies = test_agent.sample_action(obs, is_training=False)
			obs, legal_moves, reward, episode_dones = test_env.step(actions)
			done = episode_dones[0]
			total_reward += reward[0]
			episode_len += 1
		rewards.append(total_reward)
		episode_lens.append(episode_len)
	
	print("Avg reward : {}".format(sum(rewards)/100))

	experiment.save("current")


def append_to(tlist, tr, val):
	tlist[0].append(val)
	tlist[1].append([tr.episodes_done, tr.steps_done, tr.updates_done])


if __name__=="__main__":
	train_a2c_agent()


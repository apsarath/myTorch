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

from myTorch.projects.planning.blocksworld.a2c.config import *
from myTorch.rllib.a2c import A2CAgent
from myTorch.utils import MyContainer
from myTorch.utils.logging import Logger

parser = argparse.ArgumentParser(description="A2C Training")
parser.add_argument('--config', type=str, default="blocksworld_matrix", help="config name")
parser.add_argument('--base_dir', type=str, default=None, help="base directory")
parser.add_argument('--game_dir', type=str, default=None, help="game json directory")
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

	env = get_batched_env(config.env_name, config.num_env, config.seed, 
												game_dir=args.game_dir, mode="train", is_one_hot_world=config.is_one_hot_world)
	experiment.register_env(env)
	test_env = get_batched_env(config.env_name, 1, config.seed, 
							game_dir=args.game_dir, mode="valid", is_one_hot_world=config.is_one_hot_world)

	a2cnet = get_a2cnet(config.env_name, env.obs_dim, env.action_dim, use_gpu=config.use_gpu, policy_type=config.policy_type,
						one_hot=config.is_one_hot_world)

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

	test_agent = A2CAgent(a2cnet.make_inference_net(),
										optimizer,
										numpy_rng,
										ent_coef = config.ent_coef,
										vf_coef = config.vf_coef,
										discount_rate=config.discount_rate,
										grad_clip = [config.grad_clip_min, config.grad_clip_max])



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
	tr.test_num_games_finished = [[],[]]
	tr.game_level = 4
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

	obs, legal_moves = env.reset(game_level=tr.game_level)
	logger.reset_a2c_training_metrics(config.num_env, tr, config.sliding_wsize)

	for i in range(tr.iterations_done, num_iterations):
		
		print(("iterations done: {}".format(tr.iterations_done)))

		update_dict = {"log_taken_pvals":[],
					   "vvals":[],
					   "entropies":[],
					   "rewards":[],
					   "episode_dones":[]}

		for t in range(config.num_steps_per_upd):
			# TO DO : make changes to avoid passing the "dones" as a list
			actions, log_taken_pvals, vvals, entropies, pvals = agent.sample_action(obs, dones=update_dict["episode_dones"], is_training=True)
			obs, legal_moves, rewards, episode_dones = env.step(actions, game_level=tr.game_level)
			logger.track_a2c_training_metrics(episode_dones, rewards)

			update_dict["log_taken_pvals"].append(log_taken_pvals)
			update_dict["vvals"].append(vvals)
			update_dict["entropies"].append(entropies)
			update_dict["rewards"].append(my_variable(torch.from_numpy(rewards).type(torch.FloatTensor), use_gpu=config.use_gpu))
			update_dict["episode_dones"].append(my_variable(torch.from_numpy(episode_dones.astype(np.float32)).type(torch.FloatTensor), use_gpu=config.use_gpu))

		_, _, update_dict["vvals_step_plus_one"], _, _ = agent.sample_action(obs, 
																		dones=update_dict["episode_dones"], 
																		is_training=True, 
																		update_agent_state=False)
		pg_loss, val_loss, entropy_loss = agent.train_step(update_dict)

		tr.iterations_done+=1
		tr.global_steps_done = tr.iterations_done*config.num_env*config.num_steps_per_upd

		append_to(tr.pg_loss, tr, pg_loss)
		append_to(tr.val_loss, tr, val_loss)
		append_to(tr.entropy_loss, tr, entropy_loss)

		logger.log_scalar_rl("train_pg_loss", tr.pg_loss[0], config.sliding_wsize, [tr.episodes_done, tr.global_steps_done, tr.iterations_done])
		logger.log_scalar_rl("train_val_loss", tr.val_loss[0], config.sliding_wsize, [tr.episodes_done, tr.global_steps_done, tr.iterations_done])
		logger.log_scalar_rl("train_entropy_loss", tr.entropy_loss[0], config.sliding_wsize, [tr.episodes_done, tr.global_steps_done, tr.iterations_done])
		print("pg_loss : {}, val_loss : {}, entropy_loss : {}".format(pg_loss, val_loss, entropy_loss))

		if tr.iterations_done % config.test_freq == 0:
			print("Testing...")
			test_agent.a2cnet.set_params(agent.a2cnet.get_params())
			reward, episode_len, num_games_finished = inference(config, test_agent, test_env, tr.game_level)
			append_to(tr.test_reward, tr, reward)
			append_to(tr.test_episode_len, tr, episode_len)
			append_to(tr.test_num_games_finished, tr, num_games_finished)
			logger.log_scalar_rl("Test_reward", tr.test_reward[0], config.sliding_wsize, [tr.episodes_done, tr.global_steps_done, tr.iterations_done])
			logger.log_scalar_rl("Test_episode_len", tr.test_episode_len[0], config.sliding_wsize, [tr.episodes_done, tr.global_steps_done, tr.iterations_done])
			logger.log_scalar_rl("Test_num_games_finished", tr.test_num_games_finished[0], config.sliding_wsize, [tr.episodes_done, tr.global_steps_done, tr.iterations_done])
			logger.log_scalar_rl("Game_level", [tr.game_level], 1, [tr.episodes_done, tr.global_steps_done, tr.iterations_done])

			if num_games_finished >= config.game_level_threshold:
				experiment.save("best_model_{}".format(tr.game_level))
				tr.game_level += 1
 
		if math.fmod(tr.iterations_done, config.save_freq) == 0:
			experiment.save("current")
		

def inference(config, test_agent, test_env, game_level):
	obs, legal_moves = test_env.reset(game_level=game_level)
	rewards, episode_lens = [], []
	while not test_env.have_games_exhausted():
		done, total_reward, episode_len = False, 0.0 ,0.0
		test_agent.reset_agent_state(batch_size=1)
		while not done:
			actions, log_taken_pvals, vvals, entropies, pvals = test_agent.sample_action(obs, is_training=False)
			sampled_actions,_,_,_, sampled_pvals = test_agent.sample_action(obs, is_training=True, update_agent_state=False)
			#print "Arg Max action: {}, sampled action : {}".format(actions, sampled_actions)
			#print "Test pvals ",pvals
			obs, legal_moves, reward, episode_dones = test_env.step(actions)
			done = episode_dones[0]
			total_reward += reward[0]
			episode_len += 1.0
				
		rewards.append(float(total_reward))
		episode_lens.append(episode_len)
		num_games_finished = float(len([epi_len for epi_len in episode_lens if epi_len < test_env.max_episode_len]))
	return sum(rewards)/len(rewards), sum(episode_lens)/len(episode_lens), num_games_finished/len(episode_lens)

def append_to(tlist, tr, val):
		tlist[0].append(val)
		tlist[1].append([tr.episodes_done, tr.global_steps_done, tr.iterations_done])

if __name__=="__main__":
	train_a2c_agent()


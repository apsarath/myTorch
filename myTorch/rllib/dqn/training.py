#!/usr/bin/env python

import os
import math
import numpy as np
import argparse

import torch

import myTorch
from myTorch.environment import make_environment
from myTorch.utils import modify_config_params, one_hot, RLExperiment, get_optimizer
from myTorch.rllib.dqn.q_networks import *

from myTorch.rllib.dqn.config import *
from myTorch.rllib.dqn import ReplayBuffer, DQNAgent
from myTorch.utils import MyContainer
from myTorch.utils.logging import Logger

parser = argparse.ArgumentParser(description="DQN Training")
parser.add_argument('--config', type=str, default="blocksworld", help="config name")
parser.add_argument('--base_dir', type=str, default=None, help="base directory")
parser.add_argument('--config_params', type=str, default="default", help="config params to change")
parser.add_argument('--exp_desc', type=str, default="default", help="additional desc of exp")
parser.add_argument('--game_dir', type=str, default=None, help="game json directory")
args = parser.parse_args()


def train_dqn_agent():
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



	env = make_environment(config.env_name, game_dir=args.game_dir)
	env.seed(seed=config.seed)
	experiment.register_env(env)

	qnet = get_qnet(config.env_name, env.obs_dim, env.action_dim, use_gpu=config.use_gpu)

	if config.use_gpu == True:
		qnet.cuda()

	optimizer = get_optimizer(qnet.parameters(), config)

	agent = DQNAgent(qnet, 
			 		optimizer, 
			 		numpy_rng,
			 		discount_rate=config.discount_rate, 
			 		grad_clip = [config.grad_clip_min, config.grad_clip_max], 
					is_double_dqn = config.is_double_dqn,
			 		target_net_soft_update=config.target_net_soft_update,
			 		target_net_update_freq=config.target_net_update_freq,
			 		target_net_update_fraction=config.target_net_update_fraction,
			 		epsilon_start=config.epsilon_start, 
			 		epsilon_end=config.epsilon_end, 
			 		epsilon_end_t = config.epsilon_end_t, 
			 		learn_start=config.learn_start)
	experiment.register_agent(agent)


	replay_buffer = ReplayBuffer(numpy_rng, size=config.replay_buffer_size, compress=config.replay_compress)
	experiment.register_replay_buffer(replay_buffer)

	logger = None
	if config.use_tflogger==True:
		logger = Logger(logger_dir)
	experiment.register_logger(logger)

	tr = MyContainer()
	tr.train_reward = [[],[]]
	tr.train_episode_len = [[],[]]
	tr.train_loss = [[],[]]
	tr.first_qval = [[],[]]
	tr.test_reward = [[],[]]
	tr.test_episode_len = [[],[]]
	tr.test_num_games_finished = [[],[]]
	tr.iterations_done = 0
	tr.steps_done = 0
	tr.updates_done = 0
	tr.episodes_done = 0
	tr.next_target_upd = config.target_net_update_freq
	experiment.register_trainer(tr)

	if not config.force_restart:
		if experiment.is_resumable("current"):
			print("resuming the experiment...")
			experiment.resume("current")
	else:
		experiment.force_restart("current")

	for i in xrange(tr.iterations_done, config.num_iterations):
		print("iterations done: {}".format(tr.iterations_done))

		for _ in xrange(config.episodes_per_iter):
			rewards, first_qval = collect_episode(env, agent, replay_buffer, is_training=True, step=tr.steps_done)
			tr.episodes_done += 1
			tr.steps_done += len(rewards)

			epi_reward = sum(rewards)
			epi_len = len(rewards)
			append_to(tr.train_reward, tr, float(epi_reward))
			append_to(tr.train_episode_len, tr, float(epi_len))
			logger.log_scalar_rl("train_reward", tr.train_reward[0], config.sliding_wsize, [tr.episodes_done, tr.steps_done, tr.updates_done])
			logger.log_scalar_rl("train_episode_len", tr.train_episode_len[0], config.sliding_wsize, [tr.episodes_done, tr.steps_done, tr.updates_done])
			if first_qval is not None:
				append_to(tr.first_qval, tr, first_qval)
				logger.log_scalar_rl("first_qval", tr.first_qval[0], config.sliding_wsize, [tr.episodes_done, tr.steps_done, tr.updates_done])


		avg_loss = 0
		try:
			if tr.steps_done > config.learn_start:
				total_loss = 0
				for _ in xrange(config.updates_per_iter):
					minibatch = replay_buffer.sample_minibatch(batch_size = config.batch_size)
					loss = agent.train_step(minibatch)
					total_loss += loss
					tr.updates_done += 1
				avg_loss = total_loss / config.updates_per_iter
				append_to(tr.train_loss, tr, avg_loss)
				logger.log_scalar_rl("train_loss", tr.train_loss[0], config.sliding_wsize, [tr.episodes_done, tr.steps_done, tr.updates_done])
				if tr.steps_done >= tr.next_target_upd:

					agent.update_target_net()
					tr.next_target_upd += config.target_net_update_freq

		except IndexError:
			# Replay Buffer does not have enough transitions yet.
			pass

		tr.iterations_done += 1

		if tr.steps_done > config.learn_start:
			if tr.iterations_done % config.test_freq == 0:
				epi_reward = 0.0
				epi_len = 0.0
				num_games_finished = 0.0
				for _ in xrange(config.test_per_iter):
					rewards, first_qval = collect_episode(env, agent, epsilon=0.0, is_training=False)
					epi_reward += sum(rewards)
					epi_len += len(rewards)
					num_games_finished += 1.0 if epi_len < env.max_episode_len else 0.0
				epi_reward = epi_reward / config.test_per_iter
				epi_len = epi_len / config.test_per_iter
				num_games_finished = num_games_finished / config.test_per_iter
				append_to(tr.test_reward, tr, epi_reward)
				append_to(tr.test_episode_len, tr, epi_len)
				append_to(tr.test_num_games_finished, tr, num_games_finished)
				logger.log_scalar_rl("test_reward", tr.test_reward[0], config.sliding_wsize, [tr.episodes_done, tr.steps_done, tr.updates_done])
				logger.log_scalar_rl("test_episode_len", tr.test_episode_len[0], config.sliding_wsize, [tr.episodes_done, tr.steps_done, tr.updates_done])
				logger.log_scalar_rl("test_num_games_finished", tr.test_num_games_finished[0], config.sliding_wsize, [tr.episodes_done, tr.steps_done, tr.updates_done])

		if math.fmod(i+1, config.save_freq) == 0:
			experiment.save("current")
		

	experiment.save("current")


def collect_episode(env, agent, replay_buffer=None, epsilon=0, is_training=False, step=None):

	reward_list = []
	first_qval = None
	transitions = []

	obs, legal_moves = env.reset(game_level=1)
	legal_moves = format_legal_moves(legal_moves, agent.action_dim)

	episode_done = False
	episode_begin = True

	while not episode_done:

		c_step = None if not is_training else step + len(reward_list) + 1
		action, qval = agent.sample_action(obs, legal_moves, epsilon=epsilon, step=c_step, is_training=is_training)

		if episode_begin:
			first_qval = qval
			episode_begin = False

		next_obs, next_legal_moves, reward, episode_done = env.step(action)
		next_legal_moves = format_legal_moves(next_legal_moves, agent.action_dim)

		transition = {}
		transition["observations"] = obs
		transition["legal_moves"] = legal_moves
		transition["actions"] =  one_hot([action], env.action_dim)[0]
		transition["rewards"] = reward
		transition["observations_tp1"] = next_obs
		transition["legal_moves_tp1"] = next_legal_moves
		transition["pcontinues"] = 0.0 if episode_done else 1.0
		transitions.append(transition)
		reward_list.append(reward)

		obs = next_obs
		legal_moves = next_legal_moves

	if is_training:
		if replay_buffer is not None:
			for transition in transitions:
				replay_buffer.add(transition)

	return reward_list, first_qval


def format_legal_moves(legal_moves, action_dim):
	
	new_legal_moves = np.zeros(action_dim) - float("inf")
	if len(legal_moves) > 0:
		new_legal_moves[legal_moves] = 0
	return new_legal_moves

def append_to(tlist, tr, val):
	tlist[0].append(val)
	tlist[1].append([tr.episodes_done, tr.steps_done, tr.updates_done])


if __name__=="__main__":
	train_dqn_agent()


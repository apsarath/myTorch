import os
import numpy as np
import argparse

import myTorch
from myTorch.environment import make_environment
from myTorch.utils import modify_config_params, one_hot, RLExperiment, get_optimizer
from myTorch.rllib.a2c.q_networks import *

from myTorch.rllib.a2c.config import *
from myTorch.rllib.a2c import ReplayBuffer, A2CAgent
from myTorch.utils import MyContainer
from myTorch.utils.logging import Logger

parser = argparse.ArgumentParser(description="a2c Training")
parser.add_argument('--config', type=str, default="a2c", help="config name")
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

	experiment = RLExperiment(config.exp_name, train_dir)
	experiment.register_config(config)

	env = make_environment(config.env_name)
	experiment.register_env(env)
	# Introduce thing about vec_subprocess_env

	qnet = eval(config.qnet)(env.obs_dim, env.action_dim, use_gpu=config.use_gpu)

	optimizer = get_optimizer(qnet.parameters(), config)
	experiment.register_optimizer(optimizer)

	agent = A2CAgent(qnet, 
			 		optimizer, 
			 		discount_rate=config.discount_rate, 
			 		grad_clip = None,
			 		target_net_update_freq=config.target_net_update_freq,
			 		epsilon_start=config.epsilon_start, 
			 		epsilon_end=config.epsilon_end, 
			 		epsilon_end_t = config.epsilon_end_t, 
			 		learn_start=config.learn_start)
	experiment.register_agent(agent)


	logger = None
	if config.use_tflogger==True:
		logger = Logger(config.logger_dir)
	experiment.register_logger(logger)

	tr = MyContainer()
	tr.train_reward = []
	tr.train_episode_len = []
	tr.train_loss = []
	tr.first_vval = []
	#tr.test_reward = []
	#tr.test_episode_len = []
	tr.cur_iter = 0
	tr.steps_done = 0
	tr.updates_done = 0
	tr.iterations_done = 0
	tr.ep_counter = np.zeros(config.num_envs)
	tr.ep_len_count = np.zeros(config.num_envs)
	tr.cum_reward = np.zeros(config.num_envs)
	tr.new_envs = np.ones(config.num_envs, dtype=bool)
	experiment.register_trainer(tr)

	for i in xrange(tr.cur_iter, config.num_iterations):
		print("iterations done: {}".format(tr.iterations_done))

		obs, legal_move = env.reset()
		actions_list = []
		log_pvals_list = []
		vvals_list = []
		rewards_list = []
		ep_continue_list = []
		for _ in xrange(config.num_bptt_steps):
			actions, log_pvals, vvals, rewards, ep_continue_list, obs, legal_moves = train_step_prep(env, agent, obs, legal_move, \
				is_training=True, step=tr.steps_done)
			tr.iterations_done += 1
			tr.steps_done += len(rewards)
			actions_list.append(actions)
			log_pvals_list.append(log_pvals)
			vvals_list.append(vvals)
			ep_continue_list.append(ep_continue)
			rewards_list.append(rewards)
		avg_loss = 0
		total_loss = 0
		try:
			if tr.steps_done > config.learn_start:
				total_loss = agent.train_step(actions_list, log_pvals_list, vvals_list, rewards_list, ep_continue_list)
				tr.updates_done += 1
				avg_loss = total_loss / (n_env*config.num_bptt_steps)
				tr.train_loss.append(avg_loss)
				log_scalar_rl("train_loss", tr.train_loss, config.sliding_wsize, [tr.iterations_done, tr.steps_done, tr.updates_done], logger)

		except IndexError:
			# Replay Buffer does not have enough transitions yet.
			pass

		#if tr.steps_done > config.learn_start:
		#	if tr.cur_iter % config.test_freq == 0:



def train_step_prep(env, agent, obs, legal_moves, epsilon=0, is_training=False, step=None):
	
	rewards = []
	done_masks = []

	c_step = None if not is_training else step + len(reward_list_list) + 1
	actions, log_pvals, vvals = agent.sample_action(obs, legal_moves, epsilon=epsilon, step=c_step, is_training=is_training)

	next_obs, next_legal_moves, reward, episode_done = env.step(actions)
	next_legal_moves = format_legal_moves(next_legal_moves)
	#transitions = []
	ep_continue = []
	reward_list = []
	for n_env in range(config.num_envs):

		if episode_done[n_env]:
			ep_continue.append(0.0)
			tr.ep_counter[n_env] += 1
			tr.train_reward.append(tr.cum_reward[n_env])
			tr.train_episode_len.append(tr.ep_len_count[n_env])
			log_scalar_rl("first_qval", tr.first_qval, config.sliding_wsize, [tr.iterations_done, tr.steps_done, tr.updates_done], logger)
			log_scalar_rl("train_reward", tr.cum_reward[n_env], config.sliding_wsize, [tr.iterations_done, tr.steps_done, tr.updates_done], logger)
			log_scalar_rl("train_episode_len", tr.ep_len_count[n_env], config.sliding_wsize, [tr.iterations_done, tr.steps_done, tr.updates_done], logger)
			tr.ep_len_count[n_env] = 0
			tr.cum_reward[n_env] = 0
			obs[n_env], legal_move[n_env] = env[n_env].reset() #check
		else:
			ep_continue.append(1.0)
			tr.cum_reward[n_env] += reward[n_env]
			tr.ep_len_count[n_env] += 1
		transitions.append(transition)
		rewards.append(reward)

		obs[n_env] = next_obs[n_env]
		legal_moves[n_env] = next_legal_moves[n_env]

	return actions, log_pvals, vvals, rewards, ep_continue, obs, legal_moves


def format_legal_moves(legal_moves, action_dim):
	
	for n_env in range(config.num_envs):
		new_legal_moves[n_env] = np.zeros(action_dim) - float("inf")
		if len(legal_moves[n_env]) > 0:
			new_legal_moves[n_env][legal_moves[n_env]] = 0	
	return new_legal_moves


if __name__=="__main__":
	train_a2c_agent()


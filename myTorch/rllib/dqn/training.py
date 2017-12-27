import numpy as np
import argparse

import myTorch
from myTorch.utils import modify_config_params, one_hot

from myTorch.rllib.dqn.config import *


parser = argparse.ArgumentParser(description="DQN Training")
parser.add_argument('--config', type=str, default="dqn", help="config name")
parser.add_argument('--base_dir', type=str, default=None, help="base directory")
parser.add_argument('--config_params', type=str, default=None, help="config params to change")
args = parser.parse_args()


def train_dqn_agent():

	config = eval(args.config)()
	if args.config_params:
		modify_config_params(config, args.config_params)
		
	

def collect_episode(env, agent, replay_buffer=None, epsilon=0, is_training=False, step=None):

	reward_list = []
	first_qval = None
	transitions = []

	obs, legal_moves = env.reset()
	legal_moves = format_legal_moves(legal_moves)

	episode_done = False
	episode_begin = True

	while not episode_done:

		c_step = None if not is_training else step + len(reward_list) + 1		
		action, qval = agent.sample_action(obs, legal_moves, epsilon=epsilon, step=c_step, is_training=is_training)

		if episode_begin:
			first_qval = qval
			episode_begin = False

		next_obs, next_legal_moves, reward, episode_done = env.step(action)
		next_legal_moves = format_legal_moves(next_legal_moves)

		transition = {}
		transition["observations"] = obs
		transition["legal_moves"] = legal_moves
		transition["actions"] =  one_hot([action], env.action_dim)
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


def format_legal_moves(legal_moves, action_dim):
	
	new_legal_moves = np.zeros(action_dim) - float("inf")
	if len(legal_moves) > 0:
		new_legal_moves[legal_moves] = 0
	return new_legal_moves


if __name__=="__main__":
	train_dqn_agent()


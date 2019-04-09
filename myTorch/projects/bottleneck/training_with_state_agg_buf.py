#!/usr/bin/env python

import os
import math
import numpy as np
import argparse

import torch
from shutil import rmtree

import myTorch
from myTorch.environment import make_environment
from myTorch.utils import modify_config_params, one_hot, RLExperiment, get_optimizer
from myTorch.rllib.dqn.q_networks import *

from myTorch.projects.bottleneck.config import *
from myTorch.projects.bottleneck.mdp_classifier import MDPCLassifier
from myTorch.rllib.dqn import ReplayBuffer, DQNAgent
from myTorch.utils import MyContainer
from myTorch.utils.logging import Logger

parser = argparse.ArgumentParser(description="DQN with MDP Training")
parser.add_argument('--config', type=str, default="home_world", help="config name")
parser.add_argument('--base_dir', type=str, default=None, help="base directory")
parser.add_argument('--config_params', type=str, default="default", help="config params to change")
parser.add_argument('--exp_desc', type=str, default="default", help="additional desc of exp")
parser.add_argument('--run_num', type=int, default=1, help="exp run number")
args = parser.parse_args()


def train_with_state_agg_buf():
    assert(args.base_dir)
    config = eval(args.config)()

    if args.config_params != "default":
        modify_config_params(config, args.config_params)

    train_dir = os.path.join(args.base_dir, config.train_dir, config.exp_name, config.env_name,
        "{}__{}".format("default", args.exp_desc))

    logger_dir = os.path.join(args.base_dir, config.logger_dir, config.exp_name, config.env_name,
        "{}__{}_state_agg_{}".format(args.config_params, args.exp_desc, args.run_num))

    if os.path.exists(logger_dir):
        print("Starting fresh logger")
        rmtree(logger_dir)

    experiment = RLExperiment(config.exp_name, train_dir, config.backup_logger)
    #experiment.register_config(config)

    torch.manual_seed(config.seed)
    numpy_rng = np.random.RandomState(seed=config.seed)

    env = make_environment(config.env_name)
    env.seed(seed=config.seed)
    experiment.register_env(env)

    qnet = get_qnet(config.env_name, env.obs_dim, env.action_dim, config.device, state_agg=True)

    optimizer = get_optimizer(qnet.parameters(), config)

    agent = DQNAgent(qnet, 
                    optimizer, 
                    numpy_rng,
                    discount_rate=config.discount_rate, 
                    grad_clip = [config.grad_clip_min, config.grad_clip_max], 
                    target_net_soft_update=config.target_net_soft_update,
                    target_net_update_freq=config.target_net_update_freq,
                    target_net_update_fraction=config.target_net_update_fraction,
                    epsilon_start=config.epsilon_start, 
                    epsilon_end=config.epsilon_end, 
                    epsilon_end_t = config.epsilon_end_t, 
                    learn_start=config.learn_start)
    experiment.register_agent(agent)

    #assert(experiment.is_resumable("current"))
    #print("resuming the pre-mdp experiment...")
    #experiment.resume("current")


    # mdp experiment stuff
    mdp_train_dir = os.path.join(train_dir, "mdp_{}".format(args.run_num))
    mdp_experiment = GenExperiment(config.exp_name, mdp_train_dir, config.backup_logger)

    replay_buffer = ReplayBuffer(numpy_rng, size=config.replay_buffer_size, compress=config.replay_compress)
    mdp_experiment.register("state_agg_replay_buffer_modified_{}".format(config.cluster_num), replay_buffer)

    assert(mdp_experiment.is_resumable("best_model"))
    print("resuming the mdp experiment...")
    mdp_experiment.resume("best_model", input_obj_tag="state_agg_replay_buffer_modified_{}".format(config.cluster_num))

    # create classifier 
    classifier = MDPCLassifier(config.device, env.action_dim, env.obs_dim, config.cluster_num,
                                                grad_clip=[config.grad_clip_min, config.grad_clip_max]).to(config.device)

    mdp_experiment.register("classifier_{}".format(config.cluster_num), classifier)

    # load best classifier
    print("Loading Classifier {}".format(config.cluster_num))
    mdp_experiment.resume("best_model", input_obj_tag="classifier_{}".format(config.cluster_num))

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

    logger = None
    if config.use_tflogger==True:
        logger = Logger(logger_dir)
        logger.force_restart()


    for i in range(tr.iterations_done, config.num_iterations):
        print(("iterations done: {}".format(tr.iterations_done)))

        avg_loss = 0
        try:
            if 1:
                total_loss = 0
                for _ in range(config.updates_per_iter):
                    minibatch = replay_buffer.sample_minibatch(batch_size = config.batch_size)
                    loss = agent.train_step(minibatch)
                    total_loss += loss
                    tr.updates_done += 1
                avg_loss = total_loss / config.updates_per_iter
                append_to(tr.train_loss, tr, avg_loss)
                logger.log_scalar_rl("train_loss", tr.train_loss[0], config.sliding_wsize, [tr.iterations_done, tr.steps_done, tr.updates_done])
                print("train_loss : {}".format(avg_loss))
                if 1:
                    agent.update_target_net()
                    tr.next_target_upd += config.target_net_update_freq

        except IndexError:
            # Replay Buffer does not have enough transitions yet.
            pass

        tr.iterations_done += 1

        if 1:
            if tr.iterations_done % config.test_freq == 0:
                print("Testing...")
                epi_reward = 0.0
                epi_len = 0.0
                num_games_finished = 0.0
                for _ in range(config.test_per_iter):
                    rewards, first_qval = collect_episode(env, agent, classifier, config.device, epsilon=0.05, is_training=False)
                    epi_reward += sum(rewards)
                    epi_len += len(rewards)
                    num_games_finished += 1.0 if len(rewards) < env.max_episode_len else 0.0
                epi_reward = epi_reward / config.test_per_iter
                epi_len = epi_len / config.test_per_iter
                num_games_finished = num_games_finished / config.test_per_iter
                append_to(tr.test_reward, tr, epi_reward)
                append_to(tr.test_episode_len, tr, epi_len)
                append_to(tr.test_num_games_finished, tr, num_games_finished)
                logger.log_scalar_rl("test_reward", tr.test_reward[0], config.sliding_wsize, [tr.iterations_done, tr.steps_done, tr.updates_done])
                logger.log_scalar_rl("test_episode_len", tr.test_episode_len[0], config.sliding_wsize, [tr.iterations_done, tr.steps_done, tr.updates_done])
                logger.log_scalar_rl("test_num_games_finished", tr.test_num_games_finished[0], config.sliding_wsize, [tr.iterations_done, tr.steps_done, tr.updates_done])


def collect_episode(env, agent, classifier, device, replay_buffer=None, epsilon=0, is_training=False, step=None):

    def format_state_agg(obs, action=one_hot([0], env.action_dim)[0]):
        # get obs_ids
        _, obs_cluster_id, _ = classifier.predict_cluster_id_rewards({
                    "obs" : torch.from_numpy(obs[1]).type(torch.LongTensor).to(device),
                    "actions": torch.from_numpy(action).type(torch.FloatTensor).to(device)})
        obs = np.append(obs, [[0]*2,[0]*2], axis=1)
        obs[1].fill(0)
        obs[1][obs_cluster_id] = 1
        return obs
         

    reward_list = []
    first_qval = None
    transitions = []

    obs, legal_moves = env.reset()
    obs = format_state_agg(obs)
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

        reward_list.append(reward)
        obs = next_obs
        obs = format_state_agg(obs, one_hot([action], env.action_dim)[0])
        legal_moves = next_legal_moves

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
    train_with_state_agg_buf()

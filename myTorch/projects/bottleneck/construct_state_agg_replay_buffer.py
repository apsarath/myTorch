#!/usr/bin/env python

import os
import math
import numpy as np
import argparse
import time

import torch

import myTorch
from myTorch.environment import make_environment
from myTorch.utils import modify_config_params, one_hot, RLExperiment, get_optimizer, load_w2v_vectors, GenExperiment
from myTorch.rllib.dqn.q_networks import *


from sklearn.cluster import KMeans
from myTorch.projects.bottleneck.config import *
from myTorch.projects.bottleneck.mdp_classifier import MDPCLassifier
from myTorch.rllib.dqn import ReplayBuffer, DQNAgent
from myTorch.utils import MyContainer
from myTorch.utils.logging import Logger

parser = argparse.ArgumentParser(description="MDP Training")
parser.add_argument('--config', type=str, default="home_world", help="config name")
parser.add_argument('--base_dir', type=str, default=None, help="base directory")
parser.add_argument('--config_params', type=str, default="default", help="config params to change")
parser.add_argument('--exp_desc', type=str, default="default", help="additional desc of exp")
parser.add_argument('--run_num', type=int, default=1, help="exp run number")
args = parser.parse_args()

def train_mdp():
    assert(args.base_dir)
    config = eval(args.config)()
    if args.config_params != "default":
        modify_config_params(config, args.config_params)

    train_dir = os.path.join(args.base_dir, config.train_dir, config.exp_name, config.env_name, 
        "{}__{}".format(args.config_params, args.exp_desc))

    torch.manual_seed(config.seed)
    numpy_rng = np.random.RandomState(seed=config.seed)

    experiment = RLExperiment(config.exp_name, train_dir, config.backup_logger)
    replay_buffer = ReplayBuffer(numpy_rng, size=config.replay_buffer_size, compress=config.replay_compress)
    experiment.register_replay_buffer(replay_buffer)

    env = make_environment(config.env_name)
    env.seed(seed=config.seed)
    experiment.register_env(env)

    assert (experiment.is_resumable("current"))
    print("resuming the experiment...")
    experiment.resume("current")

    # Modify action dtype
    for d_id in range(len(replay_buffer.data["actions"])):
        if replay_buffer.data["actions"][d_id] is not None:
            replay_buffer.data["actions"][d_id] = replay_buffer.data["actions"][d_id].astype(np.uint8)

    # mdp experiment stuff
    mdp_train_dir = os.path.join(train_dir, "mdp_{}".format(args.run_num))
    mdp_experiment = GenExperiment(config.exp_name, mdp_train_dir, config.backup_logger)

    classifier = {}
    state_agg_replay_buffer = {}
    for num_clusters in [4,12,16,24,32,8]:
        # cluster them
        print(("Loading classifier with cluster size : ... {}".format(num_clusters)))

        # create classifier 
        classifier[num_clusters] = MDPCLassifier(env.action_dim, env.obs_dim, num_clusters, 
                                                grad_clip=[config.grad_clip_min, config.grad_clip_max]).to(config.device)

        mdp_experiment.register("classifier_{}".format(num_clusters), classifier[num_clusters])

        # load best classifier
        print("Loading Classifier {}".format(num_clusters))
        mdp_experiment.resume("best_model", input_obj_tag="classifier_{}".format(num_clusters))

        # Create new samples and create a new replay buffer.
        state_agg_replay_buffer[num_clusters] = ReplayBuffer(numpy_rng, size=config.replay_buffer_size, compress=config.replay_compress)
        mdp_experiment.register("state_agg_replay_buffer_{}".format(num_clusters), state_agg_replay_buffer[num_clusters])
        
        # Copy stuff from the old replay buffer into the new one.
        #for key in replay_buffer.data:
        #    state_agg_replay_buffer[num_clusters].data[key] = replay_buffer.data[key]
        state_agg_replay_buffer[num_clusters].write_all_data(replay_buffer.get_all_data())

        start_time = time.time()
        for d_id in range(len(replay_buffer.data["observations"])):
            if replay_buffer.data["observations"][d_id] is None:
                continue

            # get obs_ids
            _, obs_cluster_id, _ = classifier[num_clusters].predict_cluster_id_rewards({
                    "obs" : torch.from_numpy(replay_buffer.data["observations"][d_id][1]).type(torch.LongTensor).to(config.device),
                    "actions": torch.from_numpy(replay_buffer.data["actions"][d_id]).type(torch.FloatTensor).to(config.device)})

            _, obs_tp1_cluster_id, _ = classifier[num_clusters].predict_cluster_id_rewards({
                    "obs" : torch.from_numpy(replay_buffer.data["observations_tp1"][d_id][1]).type(torch.LongTensor).to(config.device),
                    "actions": torch.from_numpy(replay_buffer.data["actions"][d_id]).type(torch.FloatTensor).to(config.device)})


            obs_cluster_id = obs_cluster_id.cpu().numpy()[0]
            obs_tp1_cluster_id = obs_tp1_cluster_id.cpu().numpy()[0]
            state_agg_replay_buffer[num_clusters].data["observations"][d_id][1].fill(0)
            state_agg_replay_buffer[num_clusters].data["observations"][d_id][1][obs_cluster_id] = 1

            state_agg_replay_buffer[num_clusters].data["observations_tp1"][d_id][1].fill(0)
            state_agg_replay_buffer[num_clusters].data["observations_tp1"][d_id][1][obs_cluster_id] = 1

        print(("Done in {} secs".format(time.time()-start_time)))
        mdp_experiment.save("best_model", input_obj_tag="state_agg_replay_buffer_{}".format(num_clusters))
        

def format_legal_moves(legal_moves, action_dim):

    new_legal_moves = np.zeros(action_dim) - float("inf")
    if len(legal_moves) > 0:
        new_legal_moves[legal_moves] = 0
    return new_legal_moves

def _analyze_clusters(cluster_ids, num_clusters):
    hist = {}
    for idx in range(num_clusters):
        hist[idx] = len([i for i in cluster_ids if i == idx])
    print(("Cluster hist : \n{}\n".format(hist)))


if __name__=="__main__":
    train_mdp()

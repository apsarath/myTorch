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

	qnet = get_qnet(config.env_name, env.obs_dim, env.action_dim, config.device)

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

	assert (experiment.is_resumable("current"))
	print("resuming the experiment...")
	experiment.resume("current")

	# mdp experiment stuff
	mdp_train_dir = os.path.join(train_dir, "mdp_{}".format(args.run_num))
	mdp_experiment = GenExperiment(config.exp_name, mdp_train_dir, config.backup_logger)

	# extract states from the replay buffer.
	_, id_to_str = env.vocab
	obs_list = [ obs[1] for obs in replay_buffer.data["observations"] if obs is not None]
	raw_obs_list = [obs for obs in replay_buffer.data["observations"] if obs is not None]

	# load w2v from glove
	w2v = load_w2v_vectors(config.embedding_loc)
	
	# compute bow representations
	obs_vectors = compute_bow_representations(obs_list, w2v, id_to_str)
	
	# format data for classifier
	data_len = len(obs_list)
	print(("Replay buffer num samples available : {}".format(data_len)))
	shuffled_indices = np.arange(data_len)
	np.random.shuffle(shuffled_indices)

	
	actions = np.array([action for action in replay_buffer.data["actions"][:len(obs_list)]], dtype=np.uint8)
	rewards = np.array([reward for reward in replay_buffer.data["rewards"][:len(obs_list)]], dtype=np.uint8)

	# train data
	train_indices = shuffled_indices[:int(0.85*data_len)]
	train_obs = np.take(np.array(obs_list), train_indices, axis=0)
	train_obs = torch.from_numpy(train_obs).type(torch.LongTensor).to(config.device).detach()
	train_actions = np.take(actions, train_indices, axis=0)
	train_actions = torch.from_numpy(train_actions).type(torch.FloatTensor).to(config.device)
	train_rewards = np.take(rewards, train_indices, axis=0)
	train_rewards = torch.from_numpy(train_rewards).type(torch.FloatTensor).to(config.device)
	train_data_len = train_indices.shape[0]

	#valid data
	valid_indices = shuffled_indices[int(0.85*data_len):]
	valid_obs = np.take(np.array(obs_list), valid_indices, axis=0)
	valid_obs = torch.from_numpy(valid_obs).type(torch.LongTensor).to(config.device).detach()
	valid_actions = np.take(actions, valid_indices, axis=0)
	valid_actions = torch.from_numpy(valid_actions).type(torch.FloatTensor).to(config.device)
	valid_rewards = np.take(rewards, valid_indices, axis=0)
	valid_rewards = torch.from_numpy(valid_rewards).type(torch.FloatTensor).to(config.device)
	valid_data_len = valid_indices.shape[0]
	
	classifier = {}
	new_replay_buffer = {}
	for num_clusters in [4,12,16,24,32,8]:
		# cluster them
		print(("Clustering ... {}".format(num_clusters)))
		model = KMeans(n_clusters=num_clusters, random_state=1)
		model.fit(obs_vectors)
		cluster_ids = model.labels_
		_analyze_clusters(cluster_ids, num_clusters)
		train_cluster_ids = np.take(cluster_ids, train_indices, axis=0)
		train_cluster_ids = torch.from_numpy(train_cluster_ids).type(torch.LongTensor).to(config.device)
		valid_cluster_ids = np.take(cluster_ids, valid_indices, axis=0)
		valid_cluster_ids = torch.from_numpy(valid_cluster_ids).type(torch.LongTensor).to(config.device)

		#prepare data for the classifier
		train_minibatches = []
		for s,e in zip(list(range(0, train_data_len, config.batch_size)), list(range(config.batch_size, train_data_len, config.batch_size))):
			minibatch = {
				"obs" : train_obs[s:e],
				"actions" : train_actions[s:e],
				"rewards" : train_rewards[s:e],
				"cluster_ids" : train_cluster_ids[s:e]
			}
			train_minibatches.append(minibatch)
		
		valid_minibatches = []
		for s,e in zip(list(range(0, valid_data_len, config.batch_size)), list(range(config.batch_size, valid_data_len, config.batch_size))):
			minibatch = {  
				"obs" : valid_obs[s:e],
				"actions" : valid_actions[s:e],
				"rewards" : valid_rewards[s:e],
				"cluster_ids" : valid_cluster_ids[s:e]
			}
			valid_minibatches.append(minibatch)
	
		# train classifier 
		classifier[num_clusters] = MDPCLassifier(config.device, env.action_dim, env.obs_dim, num_clusters,
												grad_clip=[config.grad_clip_min, config.grad_clip_max]).to(config.device)

		mdp_experiment.register("classifier_{}".format(num_clusters), classifier[num_clusters])
		optimizer = get_optimizer(classifier[num_clusters].parameters(), config)
		
		# early stopping
		best_valid_acc = 0.0
		best_valid_reward_loss = float("inf")
		patience = config.patience
		for epoch_id in range(20):
			for minibatch in train_minibatches:
				classifier[num_clusters].compute_loss(minibatch, optimizer, is_training=True)
			valid_acc, reward_loss = 0.0, 0.0
			for minibatch in valid_minibatches:
				local_valid_acc, local_reward_loss = classifier[num_clusters].compute_loss(minibatch, optimizer, is_training=False)
				valid_acc += local_valid_acc
				reward_loss += local_reward_loss
			valid_acc /= len(valid_minibatches)
			reward_loss /= len(valid_minibatches)
			print(("valid acc : {}, reward_loss :{}, patience : {}".format(valid_acc, reward_loss, patience)))
			if best_valid_reward_loss > reward_loss:
				best_valid_reward_loss = reward_loss
				patience = config.patience
				mdp_experiment.save("best_model", input_obj_tag="classifier_{}".format(num_clusters))
			#if best_valid_acc < valid_acc:
			#	best_valid_acc = valid_acc
			#	patience = config.patience
			#	mdp_experiment.save("best_model", input_obj_tag="classifier_{}".format(num_clusters))
			else:
				patience -= 1
				if patience <= 0:
					break

		# load best classifier
		mdp_experiment.resume("best_model", input_obj_tag="classifier_{}".format(num_clusters))

		# Create new samples and create a new replay buffer.
		new_replay_buffer[num_clusters] = ReplayBuffer(numpy_rng, size=config.replay_buffer_size, compress=config.replay_compress)
		mdp_experiment.register("replay_buffer_{}".format(num_clusters), new_replay_buffer[num_clusters])
		obs = replay_buffer.data["observations"][0]
		transition = {}
		transition["legal_moves"] = format_legal_moves(np.arange(env.action_dim), env.action_dim)
		transition["legal_moves_tp1"] = transition["legal_moves"]
		transition["pcontinues"] = 1.0
		
		start_time = time.time()
		print(("Collecting {} samples".format(config.num_new_samples))) 
		for sample_id in range(config.num_new_samples):
			# And predict the action from the agent.
			action, _ = agent.sample_action(obs, legal_moves=None,
							epsilon=0, step=None, is_training=False)
			transition["actions"] =  one_hot([action], env.action_dim)[0]

			# get the cluster_id and reward
			_, cluster_id, reward = classifier[num_clusters].predict_cluster_id_rewards({
					"obs" : torch.from_numpy(obs[1]).type(torch.LongTensor).to(config.device).detach(),
					"actions": torch.from_numpy(transition["actions"]).type(torch.FloatTensor).to(config.device)})
			# sample the next obs from the cluster_id
			cluster_id = cluster_id.item()
			sampled_obs_id = np.random.choice([i for i,idx in enumerate(cluster_ids) if idx == cluster_id])
			next_obs = raw_obs_list[sampled_obs_id]
			
			
			transition["observations"] = obs
			transition["actions"] = one_hot([action], env.action_dim)[0]
			transition["rewards"] = reward.item()
			transition["observations_tp1"] = next_obs
			new_replay_buffer[num_clusters].add(transition)
			obs = next_obs
		print(("Done in {} secs".format(time.time()-start_time)))

		mdp_experiment.save("best_model")


def compute_bow_representations(obs_list, w2v, id_to_str):
	obs_vectors = []
	for obs in obs_list:
		obs_vector = []
		for w_id in obs:   
			if w_id in id_to_str:
				tok = id_to_str[w_id]
				if tok in w2v:
					obs_vector.append(w2v[tok])
		if len(obs_vector):   
			embedding = np.mean(np.array(obs_vector), axis=0, keepdims=False)
			embedding = embedding / np.linalg.norm(embedding)
			obs_vectors.append(embedding)
		else:          
			obs_vectors.append(np.zeros(w2v.layer1_size))
	return np.array(obs_vectors)

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

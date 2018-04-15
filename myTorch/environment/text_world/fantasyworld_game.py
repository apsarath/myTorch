import os
import string
import re
import myTorch
import numpy as np
from myTorch.environment.text_world import TelNet

class FantasyWorldGame(object):
	def __init__(self, numpy_rng, max_sent_len=100, username="default", pwd="default", ):
		# start the game server.
		self._username = username
		self._pwd = username
		self._numpy_rng = numpy_rng
		self._max_sent_len = max_sent_len
		self._str_to_id = {}
		self._id_to_str = {}
		self._init_actions_objects()

		#rewards
		self._junk_reward = -1.0
		self._default_reward = -0.01
		

		self._telnet = TelNet()
		self._telnet.send_recv("connect {} {}".format(self._username, self._pwd))
		self._telnet.send_recv("@quell")
		#print self._telnet.send_recv("@batchcommand text_sims.build")
		#print self._telnet.send_recv("start")

	def _init_actions_objects(self):
		self._actions = ['get', 'light', 'stab', 'climb', 'move', 'go']
		self._objects = ['here', 'foggy tentacles', 'east','west','north','south', 'up','down', 'northern path',
						'back to cliff', 'enter', 'leave', 'hole into cliff', 'climb the chain', 'bridge', 
						'standing archway', 'along inner wall', 'ruined gatehouse', 'castle corner', 'gatehouse',
						'courtyard', 'temple', 'stairs down', 'blue bird tomb', 'tomb of woman on horse',
						'tomb of the crowned queen', 'tomb of the shield', 'tomb of the hero', 'antechamber',
						'well', 'sign', 'tree', 'barrel', 'wall', 'obelisk', 'ghost', 'stone']
		self._unk_command_strs = ['not available', 'not find', "You can't get that.", "you cannot", "splinter is already burning", "is no way", "not uproot them"]
		self._num_rooms = 4

	def _random_teleport(self):
		#self._telnet.send_recv("@teleport Cliff by the coast")
		self._telnet.send_recv("@teleport #3")
		obs = self._telnet.send_recv("look").split(self._telnet.eom)[0]
		self._obs = self._text_to_id(obs)

	def reset(self):
		self._random_teleport()
		self._bridge_passed = False
		legal_moves = np.ones(len(self._objects) * len(self._actions))
		return (legal_moves, self._obs), self._id_to_text(self._obs)

	def _parse_text(self, text):
		available_objects = np.zeros(len(self._objects))
		reward = 0
		obs = text.split(self._telnet.eom)[0]

		for line in text.split("\n"):
			if self._telnet.eom in line:
				break

			elif "REWARD" in line:
				if not self._bridge_passed or not "REWARD_bridge" in line:
					reward += float(re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", line)[0])

				if "REWARD_bridge" in line:
					self._bridge_passed = True

			elif "Exits: " in line:
				for obj in line.split("Exits: ")[1].split(", "):
					for i, obj_inventory in enumerate(self._objects):
						if obj_inventory in obj.lower():
							available_objects[i] = 1

			elif "You see: " in line:
				for obj in line.split("You see: ")[1].split(", "):
					for i, obj_inventory in enumerate(self._objects):
						if obj_inventory in obj.lower():
							available_objects[i] = 1
			else:
				for unk_cmd_str in self._unk_command_strs:
					if unk_cmd_str in line:
						reward += self._junk_reward
			
		if reward == 0:
			reward = self._default_reward

		if not np.sum(available_objects):
			available_objects = np.ones_like(available_objects)

		return (available_objects, obs), reward

		
	def step(self, action):
		action_id = int(action) / len(self._objects)
		object_id = int(action) % len(self._objects)
		if self._actions[action_id] == "go":
			command = self._objects[object_id]
		else:
			command = "{} {}".format(self._actions[action_id], self._objects[object_id])
		#print "Command : {}".format(command) 
		(available_objects, obs), reward = self._parse_text(self._telnet.send_recv(command))

		if abs(reward - self._junk_reward) < 1e-4 or self._actions[action_id] != "go":
			(available_objects_after_look, obs_after_look), reward_after_look = \
				self._parse_text(self._telnet.send_recv("look"))
			reward += reward_after_look
			obs = obs + "." + obs_after_look if obs is not obs_after_look else obs_after_look
			available_objects = available_objects_after_look

		obs = self._text_to_id(obs)
		done = True if reward > 0.0 else False

		legal_moves = np.tile(available_objects, len(self._actions))
		return (legal_moves, obs), reward, done, self._id_to_text(obs)

	def seed(self, seed):
		self._numpy_rng = np.random.RandomState(seed)
	
	def _text_to_id(self, text):
		id_list = np.zeros(self._max_sent_len, dtype=np.int32)
		text = text.translate(None, string.punctuation)
		text = re.sub('\x1b.*?m', '', text)
		for i, word in enumerate(re.findall(r'[^,.;\s]+', text)[:self._max_sent_len]):
			word = re.sub("[^a-zA-Z]","", word)
			word = word.lower()
			if word not in self._str_to_id:
				self._str_to_id[word] = len(self._str_to_id) + 1
			id_list[i] = self._str_to_id[word]
		return id_list

	def _id_to_text(self, id_list):
		text = ""
		ivocab = self.ivocab
		for idx in id_list:
			if idx in ivocab:
				text += ivocab[idx] + " "
		return text

	@property
	def num_actions(self):
		return len(self._actions) * len(self._objects)
		
	@property
	def vocab(self):
		return self._str_to_id

	@property
	def ivocab(self):
		for k, v in list(self._str_to_id.items()):
			self._id_to_str[v] = k
		return self._id_to_str

	@property
	def num_objects(self):
		return len(self._objects)
		


if __name__=="__main__":
	numpy_rng = np.random.RandomState(1234)
	game = FantasyWorldGame(numpy_rng)
	game.reset()
	while 1:
		action = input("action: ")
		game.step(int(action))

		

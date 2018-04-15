import os
import string
import re
import myTorch
import numpy as np
from myTorch.environment.text_world import TelNet

class HomeWorldGame(object):
	def __init__(self, numpy_rng, max_sent_len=20, username="root", pwd="root", user_id=1):
		# start the game server.
		self._username = username
		self._pwd = pwd
		self._numpy_rng = numpy_rng
		self._max_sent_len = max_sent_len
		self._str_to_id = {}
		self._id_to_str = {}
		self._create_quests()
		self._translator = str.maketrans('', '', string.punctuation)

		self._telnet = TelNet()
		self._telnet.send_recv("connect {} {}".format(self._username, self._pwd))
		#print self._telnet.send_recv("@batchcommand text_sims.build")
		#print self._telnet.send_recv("start")

	def _create_quests(self):
		self._quests_mislead = ['You are not hungry.','You are not sleepy.', 'You are not bored.', 'You are not getting fat.']
		self._actions = ['eat', 'sleep', 'watch' ,'exercise', 'go']
		self._objects = ['north','south','east','west', 'tv', 'bike', 'apple', 'bed']
		self._quests_objectives = ['You are hungry.','You are sleepy.', 'You are bored.', 'You are getting fat.']
		self._quests_mislead_objectives = ['You are not hungry.','You are not sleepy.', 'You are not bored.', 'You are not getting fat.']
		self._quests = dict(list(zip(self._quests_objectives, self._actions)))
		self._mislead_quests = dict(list(zip(self._quests_mislead_objectives, self._actions)))
		self._rooms = ["living room","bedroom","garden", "kitchen"]

	def _create_random_quest(self):
		quest_ids = self._numpy_rng.choice(np.arange(len(self._quests)), size=2, replace=False)
		
		self._curr_quest_objective = "{} But now {}".format(self._quests_mislead_objectives[quest_ids[0]], \
			self._quests_objectives[quest_ids[1]])
		self._curr_quest_action = self._actions[quest_ids[1]]
		#print "Objective : {}".format(self._curr_quest_objective)

	def _random_teleport(self):
		self._telnet.send_recv("@tel limbo")
		obs = self._telnet.send_recv("start").split(self._telnet.eom)[1]
		self._obs = self._text_to_id(obs)

	def reset(self):
		self._create_random_quest()
		self._random_teleport()
		return (self._text_to_id(self._curr_quest_objective), self._obs), self._id_to_text(self._obs)
		
	def step(self, action):
		action_id = int(action) // len(self._objects)
		object_id = int(action) % len(self._objects)
		command = "{} {}".format(self._actions[action_id], self._objects[object_id])
		#print("Cmd : {}".format(command))
		data = self._telnet.send_recv(command)
		obs = data.split(self._telnet.eom)[0]
		done = False
		reward = -0.01
		if "REWARD" in data:
			obs = data.split(".")[0]
			if self._actions[action_id] == self._curr_quest_action:
				done = True
				reward = 1.0
			else:
				done = False
		elif "not available" in data or "not find" in data or "ERROR" in data:
			reward = -0.1
			obs = data.split(".")[0]
		#print(obs)
		#print self._text_to_id(obs)
		if obs == "":
			import pdb; pdb.set_trace()
		obs = self._text_to_id(obs)
		return (self._text_to_id(self._curr_quest_objective), obs), reward, done, self._id_to_text(obs)

	def seed(self, seed):
		self._numpy_rng = np.random.RandomState(seed)
	
	def _text_to_id(self, text):
		id_list = np.zeros(self._max_sent_len, dtype=np.int32)
		text = text.translate(self._translator)
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
	
	def load_vocab(self, vocab):
		self._str_to_id = vocab

	@property
	def ivocab(self):
		for k, v in list(self._str_to_id.items()):
			self._id_to_str[v] = k
		return self._id_to_str

if __name__=="__main__":
	numpy_rng = np.random.RandomState(1234)
	game = HomeWorldGame(numpy_rng)
	game.reset()
	while 1:
		action = input("action: ")
		game.step(int(action))

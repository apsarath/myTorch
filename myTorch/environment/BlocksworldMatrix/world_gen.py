import os
import json
import _pickle as pickle
import myTorch
import time
import numpy as np
from myTorch.utils import create_folder
from myTorch.environment.BlocksworldMatrix import Tower

class WorldBuilder(object):
	def __init__(self, base_dir, max_num_blocks=500, max_games_per_level=int(1e4)):
		self._base_dir = base_dir
		self._modes =  ["train","valid","test"]
		self._max_num_blocks = max_num_blocks
		self._max_games_per_level = max_games_per_level
		self._load_grounding_prob()

	def _compute_grounding_prob(self, max_num_blocks):
		start = time.time()
		print("Computing Grounding prob...")

		self._grounding_prob = {}
		for j in range(max_num_blocks):
			self._grounding_prob[(1, j)] = 1.0

		for i in range(2, max_num_blocks+1):
			for j in range(max_num_blocks):
				try:
					num = self._grounding_prob[(i-1, j)] * (i-1 + j + self._grounding_prob[(i-1, j+1)])
					den = (i - 2 + j +  self._grounding_prob[(i-1, j)])
					self._grounding_prob[(i, j)] = num / den
				except:
					continue

		print("Done Computing grounding prob. Took {} secs".format(time.time() - start))

	def _save_grounding_prob(self, max_num_blocks):
		self._compute_grounding_prob(max_num_blocks)
		loc = os.path.join(self._base_dir, "misc")
		create_folder(loc)

		file_name = os.path.join(loc, "grounding_prob_{}.pkl".format(max_num_blocks))
		with open(file_name, "wb") as f:
			pickle.dump(self._grounding_prob, f)
			print("Saved grounding_prob at {}".format(file_name))

	def _load_grounding_prob(self):
		loc = os.path.join(self._base_dir, "misc")
		if os.path.exists(loc):
			file_name = os.path.join(loc, "grounding_prob_{}.pkl".format(self._max_num_blocks))
			print("Loading grounding_prob from {}".format(file_name))
			with open(file_name, "rb") as f:
				self._grounding_prob = pickle.load(f)
		else:
			self._save_grounding_prob(self._max_num_blocks)

	def _create_game(self, game_level):

		def _create_block_list(num_towers):
			towers = [Tower(i) for i in range(2, num_towers+2)]
			num_ungrounded_towers = len(towers)

			while num_ungrounded_towers > 0:
				curr_tower = np.random.choice([tower for tower in towers if not tower.is_grounded])
				r = self._grounding_prob[(num_ungrounded_towers, num_towers-num_ungrounded_towers)]
				grounding_prob = r / (r + len(towers) - 1)
				if np.random.uniform() < grounding_prob:
					curr_tower.ground()   
				else:
					target_tower = np.random.choice([tower for tower in towers if tower != curr_tower])
					target_tower.place_on_top(curr_tower)
					towers.remove(curr_tower)

				num_ungrounded_towers -= 1

			block_list_per_tower = []
			for tower in towers:
				block_list_per_tower.append(tower.blocks)
			return block_list_per_tower
        
		game = {
			"input_towers"  : _create_block_list(num_towers=game_level),
			"target_towers" : _create_block_list(num_towers=game_level)
		}
		return game


	def create_games(self, game_level):
		loc = os.path.join(self._base_dir, "tasks/level{}".format(game_level))
		create_folder(loc)

		for mode in self._modes:
			games = []
			for game_id in range(self._max_games_per_level):
				games.append(self._create_game(game_level))
			with open(os.path.join(loc, "{}.json".format(mode)), "w") as f:
				json.dump(games, f)
		print("Created {} Games at level : {} at {}".format(self._max_games_per_level, game_level, loc))


	def load_games(self, game_level=1, mode="train"):
		games = []
		loc = os.path.join(self._base_dir, "tasks/level{}".format(game_level))
		if not os.path.exists(loc):
			self.create_games(game_level)
		print("Loading game json files at {}".format(loc))
		with open(os.path.join(loc, "{}.json".format(mode)), "r") as f:
			games =  json.load(f)
		return games


if __name__=="__main__":
	world_builder = WorldBuilder("./")
	world_builder.create_games()
	world_builder.load_games()

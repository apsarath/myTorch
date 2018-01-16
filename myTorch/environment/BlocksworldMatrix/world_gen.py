import os
import json
import myTorch
from myTorch.utils import create_folder

class WorldBuilder(object):
	def __init__(self, base_dir):
		self._base_dir = base_dir
		self._modes =  ["train","valid","test"]

	def create_games(self, game_level=1, num_games=2):
		loc = os.path.join(self._base_dir, "tasks/level{}".format(game_level))
		create_folder(loc)
	
		for mode in self._modes:
			games = []
			for game_id in range(num_games):
				games.append({ 
					"agent":{"loc":(3,0)}, 
					"input_world_blocks": [{"loc":(2,0), "color":0}],
					"target_world_blocks": [{"loc":(5,0), "color":0}] 
				})
			with open(os.path.join(loc, "{}.json".format(mode)), "w") as f:
				json.dump(games, f)
		print "Created Games at {}".format(loc)
		

	def load_games(self, game_level=1, mode="train"):
		loc = os.path.join(self._base_dir, "tasks/level{}".format(game_level))
		print "Loading game json files at {}".format(loc)
		with open(os.path.join(loc, "{}.json".format(mode)), "r") as f:
			games =  json.load(f)
		return games
			

if __name__=="__main__":
	world_builder = WorldBuilder("./")
	world_builder.create_games()
	world_builder.load_games()

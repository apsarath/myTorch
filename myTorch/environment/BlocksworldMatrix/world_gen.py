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
					"agent":{"loc":(2,0)}, 

					"input_world_blocks": [ {"loc":(0,0), "color":1, "id":2}, {"loc":(0,1), "color":2, "id":3},
											{"loc":(3,0), "color":3, "id":4}, {"loc":(3,1), "color":5, "id":6},
											{"loc":(5,0), "color":4, "id":5}, {"loc":(5,1), "color":6, "id":7}],

					"target_world_blocks": [ {"loc":(0,0), "color":2, "id":3}, {"loc":(0,1), "color":1, "id":2},
											 {"loc":(3,0), "color":5, "id":6}, {"loc":(3,1), "color":3, "id":4},
											 {"loc":(5,0), "color":6, "id":7}, {"loc":(5,1), "color":4, "id":5}]
				})
			with open(os.path.join(loc, "{}.json".format(mode)), "w") as f:
				json.dump(games, f)
		print "Created Games at {}".format(loc)
		

	def load_games(self, game_level=1, mode="train"):
		games = []
		loc = os.path.join(self._base_dir, "tasks/level{}".format(game_level))
		if os.path.exists(loc):
			print "Loading game json files at {}".format(loc)
			with open(os.path.join(loc, "{}.json".format(mode)), "r") as f:
				games =  json.load(f)
		return games
			

if __name__=="__main__":
	world_builder = WorldBuilder("./")
	world_builder.create_games()
	world_builder.load_games()

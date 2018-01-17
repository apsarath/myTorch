#!/usr/bin/env python

import os
import math
import numpy as np
from myTorch.environment import EnivironmentBase
from myTorch.environment.BlocksworldMatrix import BlocksWorld, WorldBuilder


class BlocksWorldMatrixEnv(EnivironmentBase):
    def __init__(self, game_base_dir, height=10, width=10, num_blocks=10, num_colors=1, num_steps_cutoff=50, mode="train", game_level=1, start_game_id=0) :
        self._height = height
        self._width = width
        self._num_blocks = num_blocks
        self._num_colors = num_colors 
        self._actions = {0:'left',1:'right',2:'pick',3:'drop'}
        self._legal_moves = np.array(self._actions.keys())
        self._num_steps_cutoff = num_steps_cutoff

        self._mode = mode
        self._game_level = game_level
        self._world_builder = WorldBuilder(game_base_dir)
        self._game_id = start_game_id
        self.load_games()

    def reset(self, game_level=None):

        if game_level is not None:
            self._game_level = game_level
            self._games = self._world_builder.load_games(self._game_level, self._mode)
            self._game_id = 0
            self._num_available_games = len(self._games)

        game = self._games[self._game_id]

        self._input_world = BlocksWorld(self._height, self._width, self._num_blocks, self._num_colors, is_agent_present=True)
        self._target_world = BlocksWorld(self._height, self._width, self._num_blocks, self._num_colors, is_agent_present=False)
        self._target_world.reset(blocks_info=game["target_world_blocks"])
        self._input_world.reset(blocks_info=game["input_world_blocks"],
                                agent_info=game["agent"],
                                order_look_up=self._target_world.tower.order_look_up,
                                target_height_at_loc=self._target_world.height_at_loc)
        obs = np.stack((self._input_world.as_numpy(), self._target_world.as_numpy()), axis=0)
        self._game_id = (self._game_id + 1) % self._num_available_games
        self._num_steps_done = 0
        return obs, self._legal_moves

    def step(self, action):
        self._num_steps_done += 1
        reward, done = self._input_world.update(self._actions[action])
        obs = np.stack((self._input_world.as_numpy(), self._target_world.as_numpy()), axis=0)
        if self._num_steps_done >= self._num_steps_cutoff:
            done = True
        return obs, self._legal_moves, reward, done

    def create_games(self, num_levels=1, num_games=3):
        for level in range(1,num_levels+1):
            self._world_builder.create_games(game_level=level, num_games=num_games)
        self.load_games()

    def load_games(self):
        self._games = self._world_builder.load_games(self._game_level, self._mode)
        if not len(self._games):
            self.create_games()
        self._num_available_games = len(self._games)

    @property
    def action_dim(self):
        return len(self._actions) 

    @property
    def obs_dim(self):
        return (2, self._width, self._height)

    @property
    def input_world(self):
        return self._input_world

    @property
    def target_world(self):
        return self._target_world
    
    def render(self, mode='rgb_array'):
        pass

    def seed(self, seed):
        pass

    def get_random_state(self):
        pass

    def set_random_state(self, state):
        pass

    def save(self, save_dir):
        pass

    def load(self, save_dir):
        pass

if __name__=="__main__":
    env = BlocksWorldMatrixEnv(game_base_dir="games/")
    #env.create_games()
    env.reset()
    print "Target World Down:"
    print np.flipud(np.transpose(env.target_world.as_numpy()))
    print "Input World Down:"
    print np.flipud(np.transpose(env.input_world.as_numpy()))
    print "Agent loc : {}".format(env.input_world.agent.loc)

    action_dict = {"l":0, "r":1, "p":2, "d":3}
    while True:
        action = raw_input("Action: Choose among: l,r,p,d \n")
        if action in action_dict:
            obs, _, reward, done = env.step(action_dict[action])
            print "Target World Down:"
            print np.flipud(np.transpose(obs[1]))
            print "Input World Down:"
            print np.flipud(np.transpose(obs[0]))
            print "Reward : {}, done : {}".format(reward,done)
            print "Agent loc : {}".format(env.input_world.agent.loc)

            if done:
                print "GAME OVER !!"
                obs, _ = env.reset()
                print "Target World Down:"
                print np.flipud(np.transpose(obs[1]))
                print "Input World Down:"
                print np.flipud(np.transpose(obs[0]))
                print "Agent loc : {}".format(env.input_world.agent.loc)
    
    import pdb; pdb.set_trace()

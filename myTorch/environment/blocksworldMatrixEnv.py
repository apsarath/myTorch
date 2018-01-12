#!/usr/bin/env python

import os
import math
import numpy as np
from myTorch.environment import EnivironmentBase
from myTorch.environment.BlocksworldMatrix import BlocksWorld, Block


class BlocksWorldMatrixEnv(EnivironmentBase):
    def __init__(self, height=5, width=5, num_blocks=5, num_colors=1, num_steps_cutoff=30) :
        # initialize matrix
        self._height = height
        self._width = width
        self._num_blocks = num_blocks
        self._num_colors = num_colors 
        self._actions = {0:'left',1:'right',2:'pick',3:'drop'}
        self._legal_moves = np.array(self._actions.keys())
        self._input_world = BlocksWorld(height, width, num_blocks, num_colors, is_agent_present=True)
        self._target_world = BlocksWorld(height, width, num_blocks, num_colors, is_agent_present=False)
        self._num_steps_cutoff = num_steps_cutoff
        self._num_steps_done = 0

    def reset(self):
        colors = np.random.randint(self._num_colors, size=self._num_blocks)
        self._target_world.reset([Block(i+1, color) for i, color in enumerate(colors)])
        self._input_world.reset([Block(i+1, color) for i, color in enumerate(colors)], 
                                order_look_up=self._target_world.tower.order_look_up,
                                target_height_at_loc=self._target_world.height_at_loc)
        self._obs = np.stack((self._input_world.as_numpy(), self._target_world.as_numpy()), axis=0)
        self._num_steps_done = 0
        return self._obs, self._legal_moves

    def step(self, action):
        # adjust the 2 x matrix and provide the reward if it is a legal move.
        self._num_steps_done += 1

        done = False
        reward = self._input_world.update(self._actions[action])

        self._obs = np.stack((self._input_world.as_numpy(), self._target_world.as_numpy()), axis=0)
        
        if self._actions[action] == "drop":
            if self._input_world.has_game_ended():
                done = True
        elif self._num_steps_done == self._num_steps_cutoff:
            done = True
            
        return self._obs, self._legal_moves, reward, done

    @property
    def action_dim(self):
        return len(self._actions) 

    @property
    def obs_dim(self):
        return (2, self._input_world.as_numpy().shape[0],  self._input_world.as_numpy().shape[1])

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
    env = BlocksWorldMatrixEnv(num_blocks=3, num_colors=1)
    env.reset()
    print "Target World :"
    print env.target_world
    print "Input World :"
    print env.input_world

    action_dict = {"l":0, "r":1, "p":2, "d":3}
    while True:
        action = raw_input("Action: Choose among: l,r,p,d \n")
        if action in action_dict:
            _, reward, done = env.step(action_dict[action])
            print "Target World :"
            print env.target_world
            print "Input World :"
            print env.input_world
            print "Reward : {}, done : {}".format(reward,done)

            if done:
                env.reset()
                print "Target World :"
                print env.target_world
                print "Input World :"
                print env.input_world
    
    import pdb; pdb.set_trace()

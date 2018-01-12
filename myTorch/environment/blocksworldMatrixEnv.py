#!/usr/bin/env python

import os
import math
import numpy as np
from myTorch.environment import EnivironmentBase

class Tower(object):
    def __init__(self, block_id_lookup, height_at_loc, order_look_up=None):
        self._block_id_lookup = block_id_lookup
        self._order_look_up = order_look_up
        self._height_at_loc = height_at_loc
        self._width = len(height_at_loc)
        self._ground = "ground"

    @property
    def order_look_up(self):
        return self._order_look_up

    def in_position(self, loc):
        return self._in_position[loc]

    def set_in_position_flag(self, loc):
        x, y = loc
        block_id = self._block_id_lookup[loc]
        if y > 0:
            if not self._in_position[(x,y-1)]:
                self._in_position[(x,y)] = False
            else:
                block_id_below = self._block_id_lookup[(x,y-1)]
                self._in_position[loc] = (self._order_look_up[block_id] == block_id_below)
        elif y == 0:
            self._in_position[loc] = (self._order_look_up[block_id] == self._ground)
        else:
            assert(0)

    def init_in_position_flags(self):
        self._in_position = {}

        for x in range(self._width):
            for y in range(self._height_at_loc[x]):
                block_id = self._block_id_lookup[(x,y)]
                if y == 0:
                    self._in_position[(x,y)] = (self._order_look_up[block_id] == self._ground)
                else:
                    block_id_below = self._block_id_lookup[(x,y-1)]
                    if not self._in_position[(x,y-1)]:
                        self._in_position[(x,y)] = False
                    else:
                        self._in_position[(x,y)] = (self._order_look_up[block_id] == block_id_below)

    def compute_order_look_up(self):
        self._order_look_up = {}
        for x in range(self._width):
            for y in range(self._height_at_loc[x]):
                block_id = self._block_id_lookup[(x,y)]
                if y == 0:
                    self._order_look_up[block_id] = self._ground
                else:
                    self._order_look_up[block_id] = self._block_id_lookup[(x,y-1)]
        return self._order_look_up

class BlocksWorld(object):
    def __init__(self, height=50, width=50, num_blocks=1, num_unique_blocks=1, is_agent_present=False):
        self._height = height
        self._width = width
        self._num_blocks = num_blocks
        assert(self._num_blocks < self._height * self._width)
        self._num_unique_blocks = num_unique_blocks
        self._is_agent_present = is_agent_present
        self._world = np.zeros((self._width, self._height))
        self._agent_states = ["free", "carrying"]
        self._agent_state = None
        self._agent_id = (self._num_unique_blocks + 1)

    @property
    def tower(self):
        return self._tower

    def reset(self, block_ids, order_look_up=None):
        # Create target world
        self._height_at_loc = [0]*self._width
        self._block_id_lookup = {}
        for i in range(self._num_blocks):
            block_id = block_ids[i]
            while True:
                loc = np.random.randint(self._width)
                if self._height_at_loc[loc] < self._height:
                    self._world[loc, self._height_at_loc[loc]] = block_id
                    self._block_id_lookup[(loc, self._height_at_loc[loc])] = block_id
                    self._height_at_loc[loc] += 1
                    break

        self._tower = Tower(self._block_id_lookup, self._height_at_loc, order_look_up)

        if order_look_up is None:
            self._tower.compute_order_look_up()

        self._tower.init_in_position_flags()

        if self._is_agent_present:
            self._agent_state = "free"
            while True:
                loc = np.random.randint(self._width)
                if self._height_at_loc[loc] < self._height:
                    self._world[loc, self._height_at_loc[loc]] = self._agent_id
                    self._agent_loc = (loc, self._height_at_loc[loc])
                    self._height_at_loc[loc] += 1
                    break

        return self._world

    def as_numpy(self):
        return self._world

    @property
    def agent_loc(self):
        return self._agent_loc
    
    def is_matching(self, target):
        x,y = self._agent_loc
        self._world[x, y] = 0
        is_match = np.array_equal(self._world, target.as_numpy())
        self._world[x, y] = self._agent_id
        return is_match

    def _move_agent(self, new_loc):
        curr_x, curr_y = self._agent_loc
        self._world[curr_x, curr_y] = 0
        self._height_at_loc[curr_x] -= 1
        
        new_x, new_y = new_loc
        self._world[new_x, new_y] = self._agent_id
        self._height_at_loc[new_x] += 1
        self._agent_loc = new_loc

    def _move_block(self, curr_loc, new_loc):
        block_id = self._block_id_lookup[curr_loc]
        del self._block_id_lookup[curr_loc]
        curr_x, curr_y = curr_loc
        self._world[curr_x, curr_y] = 0
        self._height_at_loc[curr_x] -= 1
        
        new_x, new_y = new_loc
        self._world[new_x, new_y] = block_id
        self._height_at_loc[new_x] += 1
        self._block_id_lookup[new_loc] = block_id

    @property
    def agent_state(self):
        return self._agent_state
        
    def update(self, action):
        print "prev agent state: {}".format(self._agent_state)
        (x, y) = self._agent_loc

        if action == "left":
            if x == 0: return
            if self._agent_state == "free":
                if self._height - self._height_at_loc[x-1] > 0:
                    self._move_agent((x-1, self._height_at_loc[x-1]))
                return
            elif self._agent_state == "carrying":
                if self._height - self._height_at_loc[x-1] > 1:
                    self._move_agent((x-1, self._height_at_loc[x-1]))
                    self._move_block((x,y+1), (x-1, self._height_at_loc[x-1]))
                return
                    
        elif action == "right":
            if x == (self._width - 1): return
            if self._agent_state == "free":
                if self._height - self._height_at_loc[x+1] > 0:
                    self._move_agent((x+1, self._height_at_loc[x+1]))
                return
            elif self._agent_state == "carrying":
                if self._height - self._height_at_loc[x+1] > 1:
                    self._move_agent((x+1, self._height_at_loc[x+1]))
                    self._move_block((x,y+1), (x+1, self._height_at_loc[x+1]))
                return

        elif action == "drop":
            if self._agent_state == "free": return
            self._move_agent((x, y+1))
            self._move_block((x, y+1),(x, y))
            self._tower.set_in_position_flag((x, y))
            self._agent_state = "free"
            return

        elif action == "pick":
            if self._agent_state == "carrying": return
            if y == 0: return
            if (x,y-1) in self._block_id_lookup:
                self._move_agent((x,y-1))
                self._move_block((x,y-1),(x,y))
                self._agent_state = "carrying"
            return

    def __str__(self):
        block_syms = ["B{}".format(i) for i in range(self._num_unique_blocks)]
        world = [ [""]*self._width for _ in range(self._height)]

        for loc, block_id in self._block_id_lookup.items():
            w, h = loc
            world[h][w] = block_syms[block_id-1]

        if self._is_agent_present:
            agent_sym = "A"
            w, h = self._agent_loc
            world[h][w] = agent_sym
        
        for row in world[::-1]: 
            print row

        print "=========== World matrix ============"
        print np.transpose(self._world)
        
        return "************************"


class BlocksWorldMatrixEnv(EnivironmentBase):
    def __init__(self, height=5, width=5, num_blocks=1, num_unique_blocks=1, 
                 max_reward = 10, min_reward = 0, num_steps_cutoff=20) :
        # initialize matrix
        self._height = height
        self._width = width
        self._num_blocks = num_blocks
        self._num_unique_blocks = num_unique_blocks 
        self._max_reward = max_reward
        self._min_reward = min_reward
        self._actions = {0:'left',1:'right',2:'pick',3:'drop'}
        self._input_world = BlocksWorld(height, width, num_blocks,num_unique_blocks, is_agent_present=True)
        self._target_world = BlocksWorld(height, width, num_blocks,num_unique_blocks, is_agent_present=False)
        self._num_steps_cutoff = num_steps_cutoff
        self._num_steps_done = 0

    def reset(self):
        block_ids = np.random.randint(1, self._num_unique_blocks+1, size=self._num_blocks)
        self._target_world.reset(block_ids)
        self._input_world.reset(block_ids, order_look_up=self._target_world.tower.order_look_up)
        self._obs = np.concatenate((self._input_world.as_numpy(), self._target_world.as_numpy()), axis=0)
        self._num_steps_done = 0
        return self._obs

    def step(self, action):
        # adjust the 2 x matrix and provide the reward if it is a legal move.
        self._num_steps_done += 1

        reward, done = self._min_reward, False

        self._input_world.update(self._actions[action])
        self._obs = np.concatenate((self._input_world.as_numpy(), self._target_world.as_numpy()), axis=0)
        
        if self._actions[action] == "drop":
            if self._input_world.is_matching(self._target_world):
                reward = self._max_reward
                done = True
        elif self._num_steps_done == self._num_steps_cutoff:
            done = True
            
        return self._obs, reward, done

    @property
    def action_dim(self):
        return len(self._actions) 

    @property
    def obs_dim(self):
        return (height, width)

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
    env = BlocksWorldMatrixEnv()
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
            print env.input_world
            print "Reward : {}, done : {}".format(reward,done)
    
    import pdb; pdb.set_trace()

#!/usr/bin/env python

import os
import math
import numpy as np
import myTorch
from myTorch.environment.BlocksworldMatrix import Agent, Tower

class BlocksWorld(object):
    def __init__(self, height=50, width=50, num_blocks=1, num_colors=1, is_agent_present=False):
        self._height = height
        self._width = width
        self._num_blocks = num_blocks
        assert(self._num_blocks < self._height * self._width)
        self._num_colors = num_colors
        self._is_agent_present = is_agent_present

    @property
    def tower(self):
        return self._tower

    @property
    def height_at_loc(self):
        return self._height_at_loc

    def reset(self, blocks, order_look_up=None, target_height_at_loc=None):
        # reset world
        self._world = np.zeros((self._width, self._height))

        self._height_at_loc = [0]*self._width
        self._block_lookup = {}
        self._blocks = blocks
        for block in self._blocks:
            while True:
                loc = np.random.randint(self._width)
                if self._height_at_loc[loc] < self._height:
                    self._world[loc, self._height_at_loc[loc]] = block.id
                    self._block_lookup[(loc, self._height_at_loc[loc])] = block
                    block.set_loc((loc, self._height_at_loc[loc]))
                    self._height_at_loc[loc] += 1
                    break

        self._tower = Tower(self._block_lookup, self._height_at_loc, order_look_up)

        if target_height_at_loc is not None:
            self._target_height_at_loc = target_height_at_loc

        if self._is_agent_present:
            self._agent = Agent(agent_id=1)
            while True:
                loc = np.random.randint(self._width)
                if self._height_at_loc[loc] < self._height:
                    self._world[loc, self._height_at_loc[loc]] = self._agent.id
                    self._agent.set_loc((loc, self._height_at_loc[loc]))
                    self._height_at_loc[loc] += 1
                    break

        return self._world

    def as_numpy(self):
        return self._world

    def is_matching(self, target):
        x,y = self._agent.loc
        self._world[x, y] = 0
        is_match = np.array_equal(self._world, target.as_numpy())
        self._world[x, y] = self._agent.id
        return is_match

    def has_game_ended(self):
        if self._num_colors > 1:
            return (self._tower.num_blocks_in_position == self._num_blocks)
        else:
            for loc in range(self._width):
                effective_height = (self._height_at_loc[loc])
                target_height = self._target_height_at_loc[loc]
                if loc == self._agent.loc[0]:
                    effective_height -= 1
                if target_height != effective_height:
                    return False
            return True

    def update(self, action):
        (x, y) = self._agent.loc

        if action == "left":
            if x == 0: return 0
            if self._height - self._height_at_loc[x-1] > 1:
                dest_loc = (x-1, self._height_at_loc[x-1])
                self._agent.move(dest_loc, self._world, self._block_lookup, self._height_at_loc)
                return 0
                    
        elif action == "right":
            if x == (self._width - 1): return 0
            if self._height - self._height_at_loc[x+1] > 1:
                dest_loc = (x+1, self._height_at_loc[x+1])
                self._agent.move(dest_loc, self._world, self._block_lookup, self._height_at_loc)
                return 0

        elif action == "pick":
            if self._agent.block is not None: return 0
            if y == 0: return 0
            if (x,y-1) in self._block_lookup:
                block = self._block_lookup[(x,y-1)]
                self._agent.pick_up_block(block, self._world, self._block_lookup)
            return 0

        elif action == "drop":
            if self._agent.block is None: return 0
            block = self._agent.block
            in_position_before_drop = block.in_position

            self._agent.drop_block(self._world, self._block_lookup)
            if y > 0:
                block.set_in_position_flag(self._tower.order_look_up, self._block_lookup[(x,y-1)])
            else:
                block.set_in_position_flag(self._tower.order_look_up)

            in_position_after_drop = block.in_position
            reward = 0
            if in_position_before_drop == True and in_position_after_drop == False:
                reward = -1
                self._tower.add_to_num_blocks_in_position(-1)
            elif in_position_before_drop == False and in_position_after_drop == True:
                reward = 1
                self._tower.add_to_num_blocks_in_position(1)

            elif self._num_colors == 1:
                reward = 0
                if (self._height_at_loc[x] - 1) == self._target_height_at_loc[x]:
                    reward = 10
                elif (self._height_at_loc[x] - 1) - self._target_height_at_loc[x] == 1:
                    reward = 0
                
            return reward

    def __str__(self):
        world = [ ["   "]*self._width for _ in range(self._height)]

        for loc, block in self._block_lookup.items():
            w, h = loc
            world[h][w] = "B{}{}".format(block.id, block.color)

        if self._is_agent_present:
            agent_sym = "AAA"
            w, h = self._agent.loc
            world[h][w] = agent_sym
        
        for row in world[::-1]: 
            print row

        if self._is_agent_present:
            if self._num_colors > 1:
                print "----------------------\n"
                for block in self._blocks:
                    print "Block : {} - in_position : {}".format(block.id, block.in_position)
            else:
                print "----------------------\n"
                print "input : ", self._height_at_loc
                print "target", self._target_height_at_loc

            print np.flipud(np.transpose(self._world))
        return "************************"

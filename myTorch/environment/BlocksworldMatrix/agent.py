#!/usr/bin/env python

import os
import math
import numpy as np

class Agent(object):
    def __init__(self, agent_id):
        self._agent_id = agent_id
        self._loc = None
        self._block = None

    def set_loc(self, loc):
        self._loc = loc

    def _swap_agent_block_loc(self, world, block_lookup):
        del block_lookup[self._block.loc]
        bx, by = self._block.loc
        ax, ay = self._loc
        
        world[bx, by] = self._agent_id
        world[ax, ay] = self._block.id
        
        self._block.set_loc((ax, ay))
        block_lookup[self._block.loc] = self._block
        self._loc = ((bx, by))
        
        

    def pick_up_block(self, block, world, block_lookup):
        self._block = block
        self._swap_agent_block_loc(world, block_lookup)

    def drop_block(self, world, block_lookup):
        self._swap_agent_block_loc(world, block_lookup)
        self._block = None

    def move(self, dest_loc, world, block_lookup, height_at_loc):
        curr_x, curr_y = self._loc
        dest_x, dest_y = dest_loc
        world[curr_x, curr_y] = 0
        world[dest_x, dest_y] = self._agent_id
        self._loc = dest_loc
        height_at_loc[curr_x] -= 1
        height_at_loc[dest_x] += 1

        if self._block is not None:
            curr_x, curr_y = self._block.loc
            del block_lookup[self._block.loc]
            dest_y += 1 

            world[curr_x, curr_y] = 0
            world[dest_x, dest_y] = self._block.id
            self._block.set_loc((dest_x, dest_y))
            block_lookup[self._block.loc] = self._block
            height_at_loc[curr_x] -= 1
            height_at_loc[dest_x] += 1

    @property
    def loc(self):
        return self._loc

    @property
    def id(self):
        return self._agent_id
    
    @property
    def block(self):
        return self._block

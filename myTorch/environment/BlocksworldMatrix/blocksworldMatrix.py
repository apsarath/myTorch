#!/usr/bin/env python

import os
import math
import numpy as np
import myTorch
from myTorch.environment.BlocksworldMatrix import Agent, Tower, Block

class BlocksWorld(object):
    def __init__(self, height=50, width=50, num_blocks=1, num_colors=1, is_agent_present=False):
        self._height = height
        self._width = width
        self._num_blocks = num_blocks
        assert(self._num_blocks < self._height * self._width)
        self._num_colors = num_colors
        self._is_agent_present = is_agent_present
        self._agent = None

    @property
    def tower(self):
        return self._tower

    @property
    def height_at_loc(self):
        return self._height_at_loc

    @property
    def agent(self):
        return self._agent

    def reset(self, blocks_info, agent_info=None, order_look_up=None, target_height_at_loc=None):
        # reset world
        self._world = np.zeros((self._width, self._height))

        self._height_at_loc = [0]*self._width
        self._block_lookup = {}
        self._blocks = []
        for i, block_info in enumerate(blocks_info):
            block = Block(block_id=i+2, color=block_info['color'])
            block.set_loc(tuple(block_info['loc']))
            loc_x, loc_y = block.loc
            assert(self._height_at_loc[loc_x] < self._height)
            self._world[loc_x, loc_y] = block.id
            self._block_lookup[block.loc] = block
            self._height_at_loc[loc_x] += 1

        self._tower = Tower(self._block_lookup, self._height_at_loc, order_look_up)

        if target_height_at_loc is not None:
            self._target_height_at_loc = target_height_at_loc

        if self._is_agent_present:
            self._agent = Agent(agent_id=1)
            self._agent.set_loc(tuple(agent_info["loc"]))
            loc_x, loc_y = self._agent.loc
            assert(self._height_at_loc[loc_x] < self._height)
            self._world[loc_x, loc_y] = self._agent.id
            self._height_at_loc[loc_x] += 1

        return self._world

    def as_numpy(self):
        return self._world

    def is_matching(self, target):
        x,y = self._agent.loc
        self._world[x, y] = 0
        is_match = np.array_equal(self._world, target.as_numpy())
        self._world[x, y] = self._agent.id
        return is_match

    def _has_game_ended(self):
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
            if x == 0: return 0, False
            if self._height - self._height_at_loc[x-1] > 1:
                dest_loc = (x-1, self._height_at_loc[x-1])
                self._agent.move(dest_loc, self._world, self._block_lookup, self._height_at_loc)
                return 0, False
                    
        elif action == "right":
            if x == (self._width - 1): return 0, False
            if self._height - self._height_at_loc[x+1] > 1:
                dest_loc = (x+1, self._height_at_loc[x+1])
                self._agent.move(dest_loc, self._world, self._block_lookup, self._height_at_loc)
                return 0, False

        elif action == "pick":
            if self._agent.block is not None: return 0, False
            if y == 0: return 0, False
            if (x,y-1) in self._block_lookup:
                block = self._block_lookup[(x,y-1)]
                self._agent.pick_up_block(block, self._world, self._block_lookup)
            return 0, False

        elif action == "drop":
            if self._agent.block is None: return 0, False
            block = self._agent.block
            picked_x, picked_y = self._agent.picked_loc
            in_position_before_drop = block.in_position

            self._agent.drop_block(self._world, self._block_lookup)
            if y > 0:
                block.set_in_position_flag(self._tower.order_look_up, self._block_lookup[(x,y-1)])
            else:
                block.set_in_position_flag(self._tower.order_look_up)

            in_position_after_drop = block.in_position
            reward = 0
            if self._num_colors > 1:
                if in_position_before_drop == True and in_position_after_drop == False:
                    reward = -1
                    self._tower.add_to_num_blocks_in_position(-1)
                elif in_position_before_drop == False and in_position_after_drop == True:
                    reward = 1
                    self._tower.add_to_num_blocks_in_position(1)

            elif self._num_colors == 1:
                if (self._height_at_loc[x] - 1) == self._target_height_at_loc[x]:
                    reward = 1
                if (self._target_height_at_loc[picked_x] - self._height_at_loc[picked_x] == 1):
                    reward = -1
                if picked_x == x:
                    reward = 0

            done = self._has_game_ended()
            return reward, done

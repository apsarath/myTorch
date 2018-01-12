#!/usr/bin/env python

import os
import math
import numpy as np

class Tower(object):
    def __init__(self, block_lookup, height_at_loc, order_look_up=None):
        self._block_lookup = block_lookup
        self._height_at_loc = height_at_loc
        self._width = len(height_at_loc)
        self._order_look_up = order_look_up
        if order_look_up is None:
            self._compute_order_look_up()
        self._init_in_position_flags()

    @property
    def order_look_up(self):
        return self._order_look_up

    def _init_in_position_flags(self):
        self._in_position = {}

        self._num_blocks_in_position = 0
        for x in range(self._width):
            for y in range(self._height_at_loc[x]):
                block = self._block_lookup[(x,y)]
                if y == 0:
                    block.set_in_position_flag(self._order_look_up)
                else:
                    block_below = self._block_lookup[(x,y-1)]
                    block.set_in_position_flag(self._order_look_up, block_below)

                if block.in_position:
                    self._num_blocks_in_position += 1

    @property
    def num_blocks_in_position(self):
        return self._num_blocks_in_position

    def add_to_num_blocks_in_position(self, num):
        self._num_blocks_in_position += num
    
    def _compute_order_look_up(self):
        self._order_look_up = {}
        for x in range(self._width):
            for y in range(self._height_at_loc[x]):
                block = self._block_lookup[(x,y)]
                if y == 0:
                    self._order_look_up[block.id] = -1
                else:
                    self._order_look_up[block.id] = self._block_lookup[(x,y-1)].color
        return self._order_look_up

#!/usr/bin/env python

import os
import math
import numpy as np

class Block(object):
    def __init__(self, block_id, color):
        self._block_id = block_id
        self._color = color
        self._loc = None
        self._prev_loc = None
        self._in_position = None
    
    def set_loc(self, loc):
        self._prev_loc = self._loc
        self._loc = loc

    def set_in_position_flag(self, order_look_up, block_below=None):
        if block_below is not None:
            if not block_below.in_position:
                self._in_position = False
            else:
                self._in_position = (order_look_up[self._block_id] == block_below.color)
        else:
            self._in_position = (order_look_up[self._block_id] == -1)

    @property
    def in_position(self):
        return self._in_position

    @property
    def loc(self):
        return self._loc

    @property
    def prev_loc(self):
        return self._prev_loc
        
    @property
    def id(self):
        return self._block_id
    
    @property
    def color(self):
        return self._color

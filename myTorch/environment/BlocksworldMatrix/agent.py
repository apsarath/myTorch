#!/usr/bin/env python

import os
import math
import numpy as np

class Agent(object):
    def __init__(self, agent_id, state="free"):
        self._agent_id = agent_id
        self._states = ["free", "carrying"]
        self._state = state
        self._loc = None

    def set_state(self, state):
        self._state = state

    def set_loc(self, loc):
        self._loc = loc

    @property
    def loc(self):
        return self._loc

    @property
    def id(self):
        return self._agent_id
    
    @property
    def state(self):
        return self._state

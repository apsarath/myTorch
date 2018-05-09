#!/usr/bin/env python

import sys
import numpy as np
from tools import my_lib
import json

class Reader(object):

    def __init__(self, config, dataset):
        self._config = config
        self._dataset = dataset
        self._data = {}
        for mode in ["train", "valid"]:
            self._data[mode] = self._dataset.get_data(mode)

    def itr_generator(self, mode, seed=400, resume_mb_id=0):
        input_data = {}
        data_len = self._data[mode]["sources"].shape[0]
        input_data["num_batches"] = int(data_len / self._config.batch_size)
        s_e_indices = list(zip(list(range(0, data_len, self._config.batch_size)), list(range(self._config.batch_size, data_len, self._config.batch_size)))) 
        mb_id = 0   
        for s, e in s_e_indices[resume_mb_id:]:
            for k in self._data[mode]:
                input_data[k] = self._data[mode][k][s:e]
            input_data["mb_id"] = resume_mb_id + mb_id
            mb_id += 1
            yield input_data

    def get_dataset(self):
        return self._dataset

#!/usr/bin/env python

import sys
import numpy as np
import pdb
import json
import os
import torch
import _pickle


class Anime(object):
    def __init__(self, config):
        self._config = config
        self._load_data(config)
        self._prep_data()
        self._split_train_valid()

    def _load_data(self, config):
        with open(os.path.join(config.base_data_path, "Data.pkl"), "rb") as f:
            self._data = _pickle.load(f)
            self._data = [self._data[idx] for idx in self._data]

        with open(os.path.join(config.base_data_path, "Vocab.pkl"), "rb") as f:
            self._id_to_str = _pickle.load(f)
            self._str_to_id = {}
            for k in self._id_to_str:
                self._str_to_id[self._id_to_str[k]] = k

    def _prep_data(self):
        
        def _get_lens(sent):
            sent_len = 0
            for w_id in sent:
                if w_id != 0:
                    sent_len += 1
            return sent_len

        src_lens = [_get_lens(self._data[idx]["context"]) for idx in range(len(self._data))]
        sorted_indices = np.argsort(src_lens)[::-1]
        self._data = [self._data[idx] for idx in sorted_indices]

        src_lens = [_get_lens(self._data[idx]["context"]) for idx in range(len(self._data))]
        target_lens = [_get_lens(self._data[idx]["response"]) for idx in range(len(self._data))]
        self._formated_data = {}
        self._formated_data["sources"] = torch.LongTensor([self._data[idx]["context"] for idx in range(len(self._data))])
        self._formated_data["targets_input"] = \
            torch.LongTensor([self._data[idx]["response"][:-1] for idx in range(len(self._data))])
        self._formated_data["targets_output"] = \
            torch.LongTensor([self._data[idx]["response"][1:] for idx in range(len(self._data))])
        
        self._formated_data["sources_len"] = torch.LongTensor(src_lens)
        self._formated_data["target_lens"] = torch.LongTensor(target_lens)

    def _split_train_valid(self):
        self._processed_data = {"train": {}, "valid" : {}}
        data_len = self._formated_data["sources_len"].shape[0]

        #train_data
        s = 0
        e = int(data_len * self._config.train_valid_split)
        for k in self._formated_data:
            self._processed_data["train"][k] = self._formated_data[k][s:e]

        #valid data
        s = e
        e = data_len
        for k in self._formated_data:
            self._processed_data["valid"][k] = self._formated_data[k][s:e]

    @property
    def data(self):
        return self._processed_data

    @property
    def raw_data(self):
        return self._formated_data
        
    @property
    def str_to_id(self):
        return self._str_to_id

    @property
    def id_to_str(self):
        return self._id_to_str

#!/usr/bin/env python

import sys
import numpy as np
import pdb
from tools import my_lib
import json
import os

class OPUS(object):

    def __init__(self, config):
        self._config = config
        self._load_and_process_data()
        self._split_train_valid()
        
    def _load_and_process_data(self):
        self._data = {}
        if self._config.toy_mode:
            with open(os.path.join(self._config.base_data_path, "t_given_s_dialogue_length2_6_toy.txt"), "r") as f:
                lines = [line for line in f]
        else:
            with open(os.path.join(self._config.base_data_path, "t_given_s_dialogue_length2_6_500000.txt"), "r") as f:
                lines = [line for line in f]

        with open(os.path.join(self._config.base_data_path, "movie_25000"), "r") as f:
            words = [word[:-1] for word in f]

        words += self._config.extra_vocab
        
        self._vocab = list(zip(words, list(range(len(words)))))
        
        id_to_str = dict([(i,w) for w,i in self._vocab])
        str_to_id = dict([(w,i) for w,i in self._vocab])
        num_id = str_to_id[self._config.num]
        eou_id = str_to_id[self._config.eou]
        go_id = str_to_id[self._config.go]
        unk_id = str_to_id[self._config.unk]
        
        def _process(utterance):
            utterance =  [int(w)-1 for w in utterance.split(" ")]
            for loc, w_id in enumerate(utterance):
                if my_lib.num_present(id_to_str[w_id]):
                    utterance[loc] = num_id
            utterance = my_lib.remove_adjacent(utterance)
            utterance = utterance[:self._config.sentence_len_cut_off-1]
            return utterance
                    
        sources, targets = [], []
        for line in lines:
            source, target = line.split("|")
            source = _process(source) + [eou_id]
            target = [go_id] + _process(target) + [eou_id]
            if len(source) < self._config.min_sent_len or len(target) < self._config.min_sent_len:
                continue
            sources.append(source)
            targets.append(target)

        def _pad_unknowns(sent_list, cut_off):
            for i, sent in enumerate(sent_list):
                if len(sent) < cut_off:
                    sent_list[i] += (cut_off - len(sent))*[unk_id]
            return np.array(sent_list)

        self._data["sources_len"] = np.array([len(sent) for sent in sources])
        self._data["targets_len"] = np.array([len(sent)-1 for sent in targets])

        self._data["sources"] = _pad_unknowns(sources, self._config.sentence_len_cut_off)
        targets = _pad_unknowns(targets, self._config.sentence_len_cut_off + 1)
        self._data["targets_input"] = targets[:,:-1]
        self._data["targets_output"] = targets[:,1:]
        self._data["targets"] = targets
        self._data["wts"] = np.zeros_like(self._data["targets_output"], dtype=np.float32)
        for i in range(self._data["targets_input"].shape[0]):
            self._data["wts"][i,:self._data["targets_len"][i]] = 1.0

    def _split_train_valid(self):
        self._processed_data = {"train": {}, "valid" : {}}
        data_len = self._data["sources_len"].shape[0]

        #train_data
        s = 0
        e = int(data_len * self._config.train_valid_split)
        for k in self._data:
            self._processed_data["train"][k] = self._data[k][s:e]

        #valid data
        s = e
        e = data_len
        for k in self._data:
            self._processed_data["valid"][k] = self._data[k][s:e]

    def get_data(self, mode):
        return self._processed_data[mode]
        
    def get_vocab(self):
        return self._vocab   

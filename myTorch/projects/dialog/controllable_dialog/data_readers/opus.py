#!/usr/bin/env python

import sys
import numpy as np
import pdb
import json
import os
import torch
import _pickle

def _num_present(input_string):
    return any(i.isdigit() for i in input_string)

def _remove_adjacent(nums):
    i = 1
    while i < len(nums):    
        if nums[i] == nums[i-1]:
            nums.pop(i)
            i -= 1  
        i += 1
    return nums

class OPUS(object):
    def __init__(self, config):
        self._config = config
        self._load_and_process_data()
        if config.act_anotation_datasets is not None:
            for act_anotation_dataset in config.act_anotation_datasets:
                self.load_acts(act_anotation_dataset)
        self._split_train_valid()
        
    def _load_and_process_data(self):
        self._data = {}
        with open(os.path.join(self._config.base_data_path, str(self._config.num_dialogs), 
                                "t_given_s_dialogue_length2_6.txt"), "r") as f:
            lines = [line for line in f]

        with open(os.path.join(self._config.base_data_path, "movie_25000"), "r") as f:
            words = [word[:-1] for word in f]

        words += [self._config.eou, self._config.go, self._config.num, self._config.pad]
        
        self._vocab = list(zip(words, list(range(len(words)))))
        
        id_to_str = dict([(i,w) for w,i in self._vocab])
        str_to_id = dict([(w,i) for w,i in self._vocab])
        self._id_to_str = id_to_str
        self._str_to_id = str_to_id
        num_id = str_to_id[self._config.num]
        eou_id = str_to_id[self._config.eou]
        go_id = str_to_id[self._config.go]
        unk_id = str_to_id[self._config.unk]
        pad_id = str_to_id[self._config.pad]
        
        def _process(utterance):
            utterance =  [int(w)-1 for w in utterance.split(" ")]
            for loc, w_id in enumerate(utterance):
                if _num_present(id_to_str[w_id]):
                    utterance[loc] = num_id
            utterance = _remove_adjacent(utterance)
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

        def _pad_sentences(sent_list, cut_off):
            for i, sent in enumerate(sent_list):
                if len(sent) < cut_off:
                    sent_list[i] += (cut_off - len(sent))*[pad_id]
            return torch.LongTensor(sent_list)

        #sort sent lens
        src_lens = [len(line) for line in sources]
        sorted_indices = np.argsort(src_lens)[::-1]
        sources = [sources[idx] for idx in sorted_indices]
        targets = [targets[idx] for idx in sorted_indices]

        # store data
        self._data["sources_len"] = torch.LongTensor([len(sent) for sent in sources])
        self._data["targets_len"] = torch.LongTensor([len(sent)-1 for sent in targets])

        self._data["sources"] = _pad_sentences(sources, self._config.sentence_len_cut_off)
        targets = _pad_sentences(targets, self._config.sentence_len_cut_off + 1)
        self._data["targets_input"] = targets[:,:-1]
        self._data["targets_output"] = targets[:,1:]
        self._data["targets"] = targets
        self._data["sources_input_lm"] = self._data["sources"][:,:-1]
        self._data["sources_output_lm"] = self._data["sources"][:,1:]
        self._data["sources_len_lm"] = torch.LongTensor([len(sent)-1 for sent in sources])

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

    def save_acts(self, tag, acts):
        with open(os.path.join(self._config.base_data_path, str(self._config.num_dialogs),
                                "{}_acts.txt".format(tag)), "wb") as f:
            _pickle.dump(acts, f)
        print("Done saving acts !")
            
    def load_acts(self, tag):
        with open(os.path.join(self._config.base_data_path, str(self._config.num_dialogs),
                                "{}_acts.txt".format(tag)), "rb") as f:
            acts = _pickle.load(f)
            self._data["{}_source_acts".format(tag)] = torch.LongTensor(acts["source"])
            self._data["{}_target_acts".format(tag)] = torch.LongTensor(acts["target"])

    @property
    def data(self):
        return self._processed_data

    @property
    def raw_data(self):
        return self._data

    @property
    def str_to_id(self):
        return self._str_to_id   

    @property
    def id_to_str(self):
        return self._id_to_str

    def num_acts(self, tag):
        return int(np.max(self._data["{}_source_acts".format(tag)].cpu().numpy()) + 1)

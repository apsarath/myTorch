#!/usr/bin/env python

import sys
import numpy as np
import pdb
import json
import os
import myTorch
from myTorch.utils import *
from ngrams import *

class OPUS(object):

    def __init__(self, config):
        self._config = config
        self._load_and_process_data()
        self._split_train_valid()
        
    def _load_and_process_data(self):
        self._data = {}
        if self._config.toy_mode:
            self._filename = os.path.join(self._config.base_data_path, "t_given_s_dialogue_length2_6_1000.txt")
        else:
            self._filename = os.path.join(self._config.base_data_path, "t_given_s_dialogue_length2_6_1000000.txt")

        print "Loading dataset from  {}".format(self._filename)
        with open(self._filename, "r") as f:
            lines = [line for line in f]

        with open(os.path.join(self._config.base_data_path, "movie_25000"), "r") as f:
            words = [word[:-1] for word in f]

        words += [self._config.eou, self._config.go, self._config.unk, self._config.num]
        
        self._vocab = zip(words, range(len(words)))
        
        id_to_str = dict([(i,w) for w,i in self._vocab])
        str_to_id = dict([(w,i) for w,i in self._vocab])
        num_id = str_to_id[self._config.num]
        eou_id = str_to_id[self._config.eou]
        go_id = str_to_id[self._config.go]
        unk_id = str_to_id[self._config.unk]

        def _remove_adjacent_unk_num(utterance, unk_id, num_id):
            if not len(utterance): return utterance

            while utterance[0] in [unk_id, num_id]:
                utterance.pop(0)
                if not len(utterance): return utterance

            i = 1
            while i < len(utterance):
                if utterance[i] in [unk_id, num_id] and utterance[i-1] in [unk_id, num_id]:
                    utterance.pop(i)
                    i -= 1
                i += 1
            return utterance
        
        def _process(utterance):
            utterance =  [int(w)-1 for w in utterance.split(" ")]
            for loc, w_id in enumerate(utterance):
                if num_present(id_to_str[w_id]):
                    utterance[loc] = num_id
            utterance = remove_adjacent(utterance)
            utterance = _remove_adjacent_unk_num(utterance, unk_id, num_id)
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

    def ngrams(self, min_length=3, max_length=4):
        with open(self._filename, "r") as f:
            ngrams = count_ngrams(f, min_length=min_length, max_length=max_length)

        id_to_str = dict([(i,w) for w,i in self._vocab])
        num = 10
        for n in sorted(ngrams):
            print('----- {} most common {}-grams -----'.format(num, n))
            for gram, count in ngrams[n].most_common(num):
                print('{0}: {1}'.format(' '.join([id_to_str[int(w)] for w in gram]), count))
                print('')

    def count_unknowns(self):
        str_to_id = dict([(w,i) for w,i in self._vocab])
        num_id = str_to_id[self._config.num]
        eou_id = str_to_id[self._config.eou]
        go_id = str_to_id[self._config.go]
        unk_id = str_to_id[self._config.unk]
        unk_num_count, total_w = 0.0, 0.0
        for utr in self._data["sources"]:
            for w in utr:
                total_w += 1.0
                if w in [unk_id, num_id]:
                      unk_num_count += 1.0
        print("Total num unk/num : {}, total words : {} in mil".format(unk_num_count/1e6, total_w/1e6))


if __name__=="__main__":
    config = MyContainer()
    config.base_data_path = "/mnt/data/chinna/data/OpenSubData"
    config.toy_mode = False
    config.train_valid_split = 0.8
    config.eou = "<eou>"
    config.go = "<go>"
    config.unk = "<unk>"
    config.num = "<num>"
    config.sentence_len_cut_off = 25
    config.min_sent_len = 6

    dataset = OPUS(config)
    dataset.ngrams(min_length=3, max_length=5)
    dataset.count_unknowns()
    id_to_str = dict([(i,w) for w,i in dataset._vocab])
    
    import pdb; pdb.set_trace()

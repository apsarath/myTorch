#!/usr/bin/env python

import sys
import numpy as np
import re
import pdb
import json
import os
import torch
from collections import Counter
import json

class Frames(object):
    def __init__(self, config):
        data_dir = os.path.join(config.data_dir, "frames/frames.json")
        self._dialogs = json.load(open(data_dir))
        self._pad = "<pad>"
        self._unk = "<unk>"
        self._other_act = "<other>"
        self._data = []
        self._read_dialogs()
        self._contruct_vocab(config.vocab_cut_off)
        self._construct_act_vocab(config.act_cut_off)
        self._preprocess_data(config.sent_cut_off)
        self._split_train_valid(config.train_valid_split)

    def _read_dialogs(self):
        count = 0
        for dialog in self._dialogs:
            for turn in dialog["turns"]:
                data = {}
                acts = turn["labels"]["acts"]
                if len(acts) > 0:
                    utt = turn["text"]
                    raw_text = re.sub('[^A-Za-z ]+', '', utt)
                    raw_text = re.sub('\s+', ' ', raw_text)
                    raw_text = re.sub('[^A-Za-z ]+', '', raw_text)
                    data["raw_text"] = raw_text.lower()
                    data["act"] = acts[0]['name']
                    count += 1
                    self._data.append(data)
    
    def _contruct_vocab(self, vocab_cut_off):
        words = []
        for data in self._data:
            words += data["raw_text"].split()

        #print("Total_words: {}".format(len(words)))
        word_count = Counter(words)
        #print("The Top {} words".format(10))
        #for word, count in word_count.most_common(10):
        #    print("{}: {}".format(word, count))

        vocab = [word for word, _ in word_count.most_common(vocab_cut_off)]
        vocab += [self._pad, self._unk]
        self._str_to_id, self._id_to_str = {}, {}
        for i,w in enumerate(vocab):
            self._str_to_id[w] = i
            self._id_to_str[i] = w

    def _construct_act_vocab(self, act_cut_off):
        act_count = Counter([data["act"] for data in self._data])
        act_vocab = [act for act, _ in act_count.most_common(act_cut_off)]
        act_vocab += [self._other_act]
        self._act_to_id, self._id_to_act = {}, {}
        for i, a in enumerate(act_vocab):
            self._act_to_id[a] = i
            self._id_to_act[i] = a

    def _text_to_id(self, text):
        id_list = []
        for word in text.split():
            if word in self._str_to_id:
                id_list.append(self._str_to_id[word])
            else:
                id_list.append(self._str_to_id[self._unk])
        return id_list

    def _preprocess_data(self, sent_cut_off):
        for data in self._data:
            data["text_id"] = self._text_to_id(data["raw_text"])

        # pad unknowns
        len_x = np.array([len(data["text_id"]) for data in self._data])
        max_len = np.max(len_x)
        for data in self._data:
            data["text_len"] = len(data["text_id"])
            if len(data["text_id"]) < max_len:
                data["text_id"] += [self._str_to_id[self._pad]]*(max_len - len(data["text_id"]))
            

        # dialog acts
        for data in self._data:
            if data["act"] in self._act_to_id:
                data["act_id"] = self._act_to_id[data["act"]]
            else:
                data["act_id"] = self._act_to_id[self._other_act]

        # shuffle ids and remove ids if text_len < config.sent_cut_off
        shuffled_ids = np.arange(len(self._data))
        np.random.shuffle(shuffled_ids)
        self._data = [self._data[idx] for idx in shuffled_ids if self._data[idx]["text_len"] > sent_cut_off]

    def _split_train_valid(self, train_valid_split):
        self._train_valid_data = {}
        total_data_len = len(self._data)

        def _prepare_data(mode, s, e):
            mode_data = self._data[s:e]
            text_lens = [data["text_len"] for data in mode_data]
            sorted_indices = np.argsort(text_lens)[::-1]
            mode_data = [mode_data[idx] for idx in sorted_indices]

            self._train_valid_data[mode] = {}
            for k in ["act_id", "text_id", "text_len"]:
                self._train_valid_data[mode][k] = torch.LongTensor([data[k] for data in mode_data])

        _prepare_data("train",s=0,e=int(0.85*total_data_len))
        _prepare_data("valid",s=int(0.85*total_data_len),e=total_data_len)

    @property
    def data(self):
        return self._train_valid_data

    @property
    def act_to_id(self):
        return self._act_to_id

    @property
    def id_to_act(self):
        return self._id_to_act
    @property
    def str_to_id(self):
        return self._str_to_id

    @property
    def id_to_str(self):
        return self._id_to_str
                

if __name__=="__main__":
    corpus = SwitchBoard()

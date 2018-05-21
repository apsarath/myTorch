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

class Cornell(object):
    def __init__(self, config):
        self._data_dir = os.path.join(config.base_data_path, "cornell/movie_lines.txt")
        self._config = config
        self._pad = "<pad>"
        self._unk = "<unk>"
        self._other_act = "<other>"
        self._data = []
        self._read_dialogs()
        self._contruct_vocab(config.vocab_cut_off)
        self._preprocess_data(config.sent_cut_off)
        self._split_train_valid()

    def _read_dialogs(self):
        count = 0
        self._utterances = []
        self._conv_ids = []
        with open(self._data_dir, "rb") as f:
            for line in f:
                utt = str(line).split(" +++$+++ ")[-1][:-3].rstrip()
                conv_id = int(str(line).split(" +++$+++ ")[0][3:])
                raw_text = re.sub('[^A-Za-z ]+', '', utt)
                raw_text = re.sub('\s+', ' ', raw_text)
                raw_text = re.sub('[^A-Za-z ]+', '', raw_text)
                self._utterances.append(raw_text.lower())
                self._conv_ids.append(conv_id)

    def _contruct_vocab(self, vocab_cut_off):
        words = []
        for line in self._utterances:
            words += line.split()

        #print("Total_words: {}".format(len(words)))
        word_count = Counter(words)
        #print("The Top {} words".format(10))
        #for word, count in word_count.most_common(10):
        #    print("{}: {}".format(word, count))

        vocab = [word for word, _ in word_count.most_common(vocab_cut_off)]
        vocab += [self._config.pad, self._config.go, self._config.unk, self._config.eou]
        self._str_to_id, self._id_to_str = {}, {}
        for i,w in enumerate(vocab):
            self._str_to_id[w] = i
            self._id_to_str[i] = w


    def _text_to_id(self, text):
        id_list = []
        for word in text.split():
            if word in self._str_to_id:
                id_list.append(self._str_to_id[word])
            else:
                id_list.append(self._str_to_id[self._config.unk])
        return id_list

    def _preprocess_data(self, sent_cut_off):
        for i, text in enumerate(self._utterances):
            self._utterances[i] = self._text_to_id(text)

        go_id = self._str_to_id[self._config.go]
        eou_id = self._str_to_id[self._config.eou]
        pad_id = self._str_to_id[self._config.pad]
        # pad unknowns
        text_lens = []
        for i, text_ids in enumerate(self._utterances):
            if len(text_ids) < self._config.sentence_len_cut_off:
                text_ids += [eou_id] + \
            [self._str_to_id[self._config.pad]]*(self._config.sentence_len_cut_off - len(text_ids))
            else:
                self._utterances[i] = self._utterances[i][:self._config.sentence_len_cut_off] + [eou_id]


        self._utterances = [ [go_id] + text_ids for text_ids in self._utterances]

        self._sources, self._targets = [], []
        for i in range(1, len(self._utterances)):
            if self._conv_ids[i-1] - self._conv_ids[i] == 1:
                self._sources.append(self._utterances[i][1:])
                self._targets.append(self._utterances[i-1])

        def _sent_len(text_ids):
            sent_len = 0
            for w_id in text_ids:
                if w_id != pad_id:
                    sent_len += 1
            return sent_len
        
        sources_lens = [_sent_len(text_ids) for text_ids in self._sources]

        sorted_indices = np.argsort(sources_lens)[::-1]
        self._sources = [self._sources[idx] for idx in sorted_indices]
        self._targets = np.array([self._targets[idx] for idx in sorted_indices])

        self._sources_lens = [_sent_len(text_ids) for text_ids in self._sources]
        self._targets_lens = [_sent_len(text_ids) for text_ids in self._targets]

        self._data = {}
        self._data["sources"] = torch.LongTensor(self._sources)
        self._data["targets_input"] = torch.LongTensor(self._targets[:,:-1])
        self._data["targets_output"] = torch.LongTensor(self._targets[:,1:])
        self._data["sources_len"] = torch.LongTensor(self._sources_lens)

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
                

if __name__=="__main__":
    corpus = SwitchBoard()

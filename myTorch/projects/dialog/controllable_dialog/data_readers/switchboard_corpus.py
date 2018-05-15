#!/usr/bin/env python

import sys
import numpy as np
import re
import pdb
import json
import os
import torch
from collections import Counter
from myTorch.projects.dialog.controllable_dialog.data_readers.switchboard.swda import CorpusReader

class SwitchBoard(object):
    def __init__(self, vocab_cut_off=10000):
        curr_dir = os.getcwd()
        data_dir = os.path.join(os.getcwd(), "switchboard/swda/")
        self._corpus = CorpusReader(data_dir)
        self._pad = "<pad>"
        self._data = []
        self._read_transcripts()
        self._contruct_vocab(vocab_cut_off)
        self._construct_act_vocab()
        self._preprocess_data()

    def _read_transcripts(self):
        data = {}
        for transcript in self._corpus.iter_transcripts():
            for utt in transcript.utterances:
                data = {}
                raw_text = re.sub('\s+', ' ', utt.text)
                raw_text = re.sub('[^A-Za-z ]+', '', raw_text)
                data["raw_text"] = raw_text.lower()
                data["act"] = utt.act_tag
            self._data.append(data)
    
    def _contruct_vocab(self, vocab_cut_off):
        words = [self._pad]
        for data in self._data:
            words += data["raw_text"].split()

        #print("Total_words: {}".format(len(words)))
        word_count = Counter(words)
        #print("The Top {} words".format(10))
        for word, count in word_count.most_common(10):
            print("{}: {}".format(word, count))

        vocab = [word for word, _ in word_count.most_common(vocab_cut_off)]
        self._str_to_id, self._id_to_str = {}, {}
        for i,w in enumerate(vocab):
            self._str_to_id[w] = i
            self._id_to_str[i] = w

    def _construct_act_vocab(self):
        act_count = Counter([data["act"] for data in self._data])
        act_vocab = [act for act, _ in act_count.most_common(100)]
        self._act_to_id, self._id_to_act = {}, {}
        for i, a in enumerate(act_vocab):
            self._act_to_id[a] = i
            self._id_to_act[i] = a

    def _text_to_id(self, text):
        id_list = []
        for word in text.split():
            id_list.append(self._str_to_id[word])
        return id_list

    def _preprocess_data(self):
        for data in self._data:
            data["text_id"] = self._text_to_id(data["raw_text"])

        # pad unknowns
        len_x = np.array([len(data["text_id"]) for data in self._data])
        max_len = np.max(len_x)
        for data in self._data:
            if len(data["text_id"]) < max_len:
                data["text_id"] += [self._str_to_id[self._pad]]*(max_len - len(data["text_id"]))

        # dialog acts
        for data in self._data:
            data["act_id"] = self._act_to_id[data["act"]]

    @property
    def data(self):
        return self._data

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

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
from myTorch.utils import load_w2v_vectors
import _pickle

class Twitter(object):
    def __init__(self, config):
        self._data_dir = os.path.join(config.base_data_path, "chat.txt")
        self._config = config
        self._pad = "<pad>"
        self._unk = "<unk>"
        self._other_act = "<other>"
        self._data = []
        self._read_dialogs()
        self._contruct_vocab(config.vocab_cut_off)
        self._preprocess_data(config.sent_cut_off)
        if config.act_anotation_datasets is not None:
            for act_anotation_dataset in config.act_anotation_datasets:
                self.load_acts(act_anotation_dataset)
        self._split_train_valid()
        self._pretrained_embeddings(config.embedding_loc)

    def _read_dialogs(self):
        count = 0
        self._utterances = []
        
        with open(self._data_dir, "rb") as f:
            for line in f:
                utt = str(line.decode('UTF-8')).rstrip()
                raw_text = re.sub('[^A-Za-z ]+', '', utt)
                raw_text = re.sub('\s+', ' ', raw_text)
                raw_text = re.sub('[^A-Za-z ]+', '', raw_text)
                self._utterances.append(raw_text.lower())
                if len(self._utterances) >= self._config.num_dialogs*2:
                    break

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
            if len(text_ids) < sent_cut_off:
                text_ids += [eou_id] + \
            [self._str_to_id[self._config.pad]]*(sent_cut_off - len(text_ids))
            else:
                self._utterances[i] = self._utterances[i][:sent_cut_off] + [eou_id]


        self._utterances = [ [go_id] + text_ids for text_ids in self._utterances]

        sources, targets = [], []
        for i in range(0, len(self._utterances), 2):
                sources.append(self._utterances[i][1:])
                targets.append(self._utterances[i+1])

        def _sent_len(text_ids):
            sent_len = 0
            for w_id in text_ids:
                if w_id != pad_id:
                    sent_len += 1
            return sent_len
        
        sources_len = [_sent_len(text_ids) for text_ids in sources]
        targets_lens = [_sent_len(text_ids) for text_ids in targets]

        self._sources, self._targets = [],[]
        for idx in range(len(sources_len)):
            if sources_len[idx] > self._config.min_sent_len and targets_lens[idx] > self._config.min_sent_len:
                self._sources.append(sources[idx])
                self._targets.append(targets[idx])

        self._sources_len = [_sent_len(text_ids) for text_ids in self._sources]

        #sorted_indices = np.argsort(sources_lens)[::-1]
        #self._sources = [self._sources[idx] for idx in sorted_indices]
        self._targets = np.array(self._targets)

        #self._sources_lens = [_sent_len(text_ids) for text_ids in self._sources]
        #self._targets_lens = [_sent_len(text_ids) for text_ids in self._targets]

        self._data = {}
        self._data["sources"] = self._sources
        self._data["sources_target"] = [source[1:] + [pad_id] for source in self._sources]
        self._data["targets_input"] = self._targets[:,:-1]
        self._data["targets_output"] = self._targets[:,1:]
        self._data["sources_len"] = self._sources_len

    def _split_train_valid(self):
        self._processed_data = {"train": {}, "valid" : {}}
        data_len = len(self._data["sources_len"])

        def _sort_data(data):
            sorted_indices = np.argsort(data["sources_len"])[::-1]
            for k in data:
                data[k] = torch.LongTensor([data[k][idx] for idx in sorted_indices])

        #train_data
        s = 0
        e = int(data_len * self._config.train_valid_split)
        for k in self._data:
            self._processed_data["train"][k] = self._data[k][s:e]
        _sort_data(self._processed_data["train"])

        #valid data
        s = e
        e = data_len
        for k in self._data:
            self._processed_data["valid"][k] = self._data[k][s:e]
        _sort_data(self._processed_data["valid"])

    def _pretrained_embeddings(self, loc):
        if loc == 'None':
            self._embeddings = None
            return
        w2v = load_w2v_vectors(loc)
        self._embeddings = np.zeros((len(self._str_to_id), w2v.layer1_size), dtype=np.float32)
        num_avail = 0
        for word, word_id in self._str_to_id.items():
            if word in w2v:
                num_avail += 1
                self._embeddings[word_id] = w2v[word]
        print("Num loaded from pretrained : {} out of {}".format(num_avail, len(self._str_to_id)))

    def save_acts(self, tag, acts):
        with open(os.path.join(self._config.base_data_path,"{}_acts.txt".format(tag)), "wb") as f:
            _pickle.dump(acts, f)
        print("Done saving acts !")

    def load_acts(self, tag):
        with open(os.path.join(self._config.base_data_path, "{}_acts.txt".format(tag)), "rb") as f:
            acts = _pickle.load(f)
            self._data["{}_source_acts".format(tag)] = acts["source"]
            self._data["{}_target_acts".format(tag)] = acts["target"]

    def _generic_responses(self):
        self._generic_responses_text = [
            "oh my god", "i don t know",
            "i am not sure",
            "i don t think that is a good idea",
            "i am not sure that is a good idea"]

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

    @property
    def pretrained_embeddings(self):
        return self._embeddings

    def num_acts(self, tag):
        return int(np.max(np.array(self._data["{}_source_acts".format(tag)])) + 1)
    
    def padded_generic_responses(self):
        return (self._generic_responses_input, self._generic_responses_output)


if __name__=="__main__":
    corpus = SwitchBoard()

#!/usr/bin/env python

import sys
import numpy as np
import pdb
import json
import os
import torch
from myTorch.projects.dialog.controllable_dialog.data_readers.switchboard.swda import CorpusReader

class SwitchBoard(object):
    def __init__(self):
        curr_dir = os.getcwd()
        data_dir = os.path.join(os.getcwd(), "switchboard/swda/")
        self._corpus = CorpusReader(data_dir)
        self._read_transcripts()
        self._data = []

    def _read_transcripts(self):
        data = {}
        for transcript in self._corpus.iter_transcripts():
            for utt in transcript.utterances:
                data = {}
                data["text"] = utt.text
                data["act"] = utt.act_tag
    
    def 
                
                

if __name__=="__main__":
    corpus = SwitchBoard()


        
        
